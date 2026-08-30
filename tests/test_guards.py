"""Tests for the two guards that stand between a bug and the daily API allowance.

Neither budget.py nor cache.py had ever been executed before these were written.
Under a 200-call/day ceiling, a fault in either is expensive in wall-clock days,
so they are tested offline and in full before any real call is spent.

Run: python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.budget import BudgetExceeded, DailyBudget  # noqa: E402
from tacit.cache import ResponseCache, cache_key  # noqa: E402


class TestDailyBudget(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "budget_test.json"

    def tearDown(self):
        self._tmp.cleanup()

    def budget(self, limit=10):
        return DailyBudget("test", limit, path=self.path)

    def test_spend_decrements_and_reports_remaining(self):
        b = self.budget(10)
        self.assertEqual(b.remaining(), 10)
        self.assertEqual(b.spend(1), 9)
        self.assertEqual(b.used(), 1)
        self.assertEqual(b.remaining(), 9)

    def test_refuses_rather_than_exceeds(self):
        b = self.budget(3)
        b.spend(3)
        with self.assertRaises(BudgetExceeded):
            b.spend(1)
        # The refused call must not have been counted.
        self.assertEqual(b.used(), 3)
        self.assertEqual(b.remaining(), 0)

    def test_refuses_oversized_batch_atomically(self):
        """A batch that would exceed must spend nothing, not partially fill."""
        b = self.budget(10)
        b.spend(8)
        with self.assertRaises(BudgetExceeded):
            b.spend(5)
        self.assertEqual(b.used(), 8, "partial spend on a refused batch")

    def test_persists_across_instances(self):
        """State must survive a process restart, or a crash-loop re-spends the day."""
        self.budget(10).spend(4)
        self.assertEqual(self.budget(10).used(), 4)

    def test_reserve_shrinks_effective_limit(self):
        """IEEEXplore(reserve=N) is implemented as a smaller limit; verify the shape."""
        full, reserved = self.budget(200), DailyBudget("test", 200 - 50, path=self.path)
        self.assertEqual(full.remaining(), 200)
        self.assertEqual(reserved.remaining(), 150)

    def test_counts_are_per_utc_day(self):
        b = self.budget(10)
        b.spend(6)
        state = json.loads(self.path.read_text())
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(state[today], 6)
        # A different day's usage must not count against today.
        state["2020-01-01"] = 999
        self.path.write_text(json.dumps(state))
        self.assertEqual(b.used(), 6)
        self.assertEqual(b.remaining(), 4)

    def test_prunes_history_but_keeps_today(self):
        b = self.budget(10)
        state = {f"2020-01-{d:02d}": 1 for d in range(1, 29)}
        self.path.write_text(json.dumps(state))
        b.spend(1)
        kept = json.loads(self.path.read_text())
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        self.assertIn(today, kept, "pruning must never drop today's count")
        self.assertLessEqual(len(kept), 30)

    def test_corrupt_state_does_not_crash_but_fails_open(self):
        """Documents current behaviour: unreadable state resets the count to zero.

        This fails *open* (toward spending) rather than closed. Acceptable only
        because the file is written atomically-ish and locally; if this ever moves
        to shared storage, revisit.
        """
        self.path.write_text("{not json")
        self.assertEqual(self.budget(10).used(), 0)


class TestResponseCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def cache(self):
        return ResponseCache("ns", root=self.root)

    def test_roundtrip(self):
        c = self.cache()
        self.assertIsNone(c.get("https://x/?a=1"))
        c.put("https://x/?a=1", {"total_records": 7})
        self.assertEqual(c.get("https://x/?a=1"), {"total_records": 7})

    def test_key_ignores_credentials(self):
        """A rotated key must not invalidate the cache and re-spend the budget."""
        a = cache_key("https://x/search?q=1&apikey=AAA")
        b = cache_key("https://x/search?q=1&apikey=BBB")
        self.assertEqual(a, b)
        self.assertEqual(a, cache_key("https://x/search?q=1&api_key=CCC".replace("api_key", "apikey")))

    def test_key_scrubs_openalex_style_param(self):
        self.assertEqual(
            cache_key("https://x/works?filter=f&api_key=AAA"),
            cache_key("https://x/works?filter=f&api_key=BBB"),
        )

    def test_key_still_separates_real_queries(self):
        self.assertNotEqual(cache_key("https://x/?year=1995"), cache_key("https://x/?year=1996"))

    def test_cached_payload_is_not_shared_mutable_state(self):
        c = self.cache()
        c.put("https://x/?a=1", {"articles": [{"n": 1}]})
        first = c.get("https://x/?a=1")
        first["articles"].append({"n": 2})
        self.assertEqual(len(c.get("https://x/?a=1")["articles"]), 1)

    def test_stats(self):
        c = self.cache()
        c.put("https://x/?a=1", {"k": "v"})
        c.put("https://x/?a=2", {"k": "v"})
        self.assertEqual(c.stats()["entries"], 2)
        self.assertGreater(c.stats()["bytes"], 0)


if __name__ == "__main__":
    unittest.main()


class TestCanonicalUrl(unittest.TestCase):
    """The cache key must not depend on the order params were written in.

    Regression test: params are passed as kwargs, so the same logical query can be
    spelled in several orders. Before canonical_url they produced distinct cache
    keys and were therefore paid for more than once.
    """

    def setUp(self):
        from tacit.http import ApiClient
        self.canon = ApiClient.canonical_url

    def test_param_order_does_not_change_the_key(self):
        a = self.canon("https://api", {"publication_title": "ICRA", "publication_year": "2015"})
        b = self.canon("https://api", {"publication_year": "2015", "publication_title": "ICRA"})
        self.assertEqual(a, b)
        self.assertEqual(cache_key(a), cache_key(b))

    def test_different_queries_still_differ(self):
        a = self.canon("https://api", {"publication_year": "2015"})
        b = self.canon("https://api", {"publication_year": "2016"})
        self.assertNotEqual(cache_key(a), cache_key(b))


class TestCredentialScrubbing(unittest.TestCase):
    """The key is a query parameter, so it appears in every URL — and so in every
    error message that quotes one. Regression test: it must never survive into a
    message a human or a log file will see."""

    def test_scrub_url_removes_both_param_spellings(self):
        from tacit.cache import scrub_url
        secret = "s3cr3tkeyvalue"
        for spelling in ("apikey", "api_key", "APIKEY"):
            url = f"https://api/search?q=1&{spelling}={secret}&format=json"
            self.assertNotIn(secret, scrub_url(url))
            self.assertIn("REDACTED", scrub_url(url))

    def test_scrub_preserves_the_rest_of_the_query(self):
        from tacit.cache import scrub_url
        out = scrub_url("https://api/search?publication_year=1995&apikey=SECRET&format=json")
        self.assertIn("publication_year=1995", out)
        self.assertIn("format=json", out)
        self.assertNotIn("SECRET", out)


class TestOpenAlexCreditModel(unittest.TestCase):
    """OpenAlex meters credits that scale with page size, not calls. Budgeting by
    call count under-counts a paged harvest ~5x. Values verified against the live
    API's x-ratelimit headers on 2026-08-30."""

    def test_credits_match_observed_costs(self):
        from tacit.openalex import credits_for
        self.assertEqual(credits_for(1), 1)      # observed $0.0001 = 1 credit
        self.assertEqual(credits_for(200), 10)   # observed $0.0010 = 10 credits
        self.assertEqual(credits_for(100), 5)
        self.assertEqual(credits_for(25), 2)

    def test_never_free(self):
        from tacit.openalex import credits_for
        self.assertEqual(credits_for(0), 1)

    def test_full_corpus_estimate_exceeds_one_free_day(self):
        """The headline number: a 65k-work pull does not fit in a day's free quota."""
        from tacit.openalex import FREE_DAILY_CREDITS, credits_for
        calls = -(-65_000 // 100)
        total = calls * credits_for(100)
        self.assertGreater(total, FREE_DAILY_CREDITS)
        self.assertAlmostEqual(total / FREE_DAILY_CREDITS, 3.25, places=2)


class TestQuotaExhaustedNotSleptThrough(unittest.TestCase):
    def test_long_retry_after_raises_instead_of_sleeping(self):
        """A 22-hour Retry-After must not become a 22-hour sleep."""
        import tacit.http as H

        class FakeResp:
            status_code = 429
            headers = {"Retry-After": "79803"}
            text = '{"error":"Rate limit exceeded"}'

        class FakeSession:
            def get(self, *a, **k):
                return FakeResp()

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        client = H.ApiClient.__new__(H.ApiClient)
        client.cache = ResponseCache("t", root=Path(tmp.name))
        client.budget = DailyBudget("t", 10, path=Path(tmp.name) / "b.json")
        client.min_interval = 0
        client.dry_run = False
        client.timeout = 5
        client.max_retry_wait = 120.0
        client.session = FakeSession()
        client.provider_remaining = None
        client._last_call = 0.0
        client.planned, client.hits, client.misses = [], 0, 0

        with self.assertRaises(H.QuotaExhausted):
            client.get("https://api", {"per-page": 1})


class TestCrossrefVenueClassifier(unittest.TestCase):
    """Container titles are the only venue handle Crossref gives us, and they are
    inconsistent across four decades. Strings below are real, taken from live
    records during the Phase 0 probe."""

    def setUp(self):
        from tacit.crossref import classify_container
        self.c = classify_container

    def test_accepts_real_icra_containers(self):
        for s in [
            "2015 IEEE International Conference on Robotics and Automation (ICRA)",
            "2022 International Conference on Robotics and Automation (ICRA)",
            "Proceedings 2002 IEEE International Conference on Robotics and Automation (Cat. No.02CH37292)",
            "1993 IEEE International Conference on Robotics and Automation",
        ]:
            self.assertEqual(self.c(s), "ICRA", s)

    def test_accepts_real_iros_containers(self):
        for s in [
            "IEEE/RSJ International Conference on Intelligent Robots and Systems",
            "Proceedings 1995 IEEE/RSJ International Conference on Intelligent Robots and Systems",
            "2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)",
        ]:
            self.assertEqual(self.c(s), "IROS", s)

    def test_rejects_confusable_venues(self):
        """These leaked in before the exclusions were tightened — ~7% of 2018."""
        for s in [
            "IEEE Robotics and Automation Letters",
            "IEEE Transactions on Robotics and Automation",
            "International Journal of Robotics and Automation",
            "2018 International Conference on Robotics and Automation Engineering",   # ICRAE
            "2018 International Conference on Robotics and Automation Sciences",      # ICRAS
            "2018 International Conference on Mechatronics, Robotics and Automation", # ICMRA
            "2022 17th International Conference on Control, Automation, Robotics and Vision",
            "2020 5th International Conference on Automation, Control and Robotics Engineering",
            "17th International Symposium on Automation and Robotics in Construction",
        ]:
            self.assertIsNone(self.c(s), s)

    def test_handles_missing_container(self):
        self.assertIsNone(self.c(None))
        self.assertIsNone(self.c(""))


class TestCrossrefYear(unittest.TestCase):
    def test_prefers_published_over_deposit_date(self):
        """A 2002 paper deposited in 2003 must date to 2002, not 2003."""
        from tacit.crossref import work_year
        self.assertEqual(work_year({
            "published": {"date-parts": [[2002]]},
            "created": {"date-parts": [[2003, 6, 25]]},
        }), 2002)

    def test_falls_back_through_date_fields(self):
        from tacit.crossref import work_year
        self.assertEqual(work_year({"issued": {"date-parts": [[1995]]}}), 1995)
        self.assertIsNone(work_year({}))
        self.assertIsNone(work_year({"published": {"date-parts": [[]]}}))


class TestLineageContinuity(unittest.TestCase):
    """The properties docs/10 claims: no cut point, distance graded not gated,
    edge strength carried, consortium papers not allowed to dominate."""

    def setUp(self):
        from tacit import lineage
        self.L = lineage
        # A path: origin - a - b - c, so distance is unambiguous.
        papers = [
            {"authors": [{"name": "origin"}, {"name": "a"}]},
            {"authors": [{"name": "a"}, {"name": "b"}]},
            {"authors": [{"name": "b"}, {"name": "c"}]},
        ]
        self.g = lineage.build_coauthor_graph(papers)

    def test_distance_is_graded_not_gated(self):
        """The whole point: distance 3 scores lower than 2, but is NOT zero."""
        a = self.L.proximity(self.g, ["origin"], ["a"])
        b = self.L.proximity(self.g, ["origin"], ["b"])
        c = self.L.proximity(self.g, ["origin"], ["c"])
        self.assertGreater(a, b)
        self.assertGreater(b, c)
        self.assertGreater(c, 0.0, "distance-3 author was gated out, not graded")

    def test_no_author_is_classified(self):
        """Scores are continuous, not a two-valued membership."""
        vals = {round(self.L.proximity(self.g, ["origin"], [n]), 12)
                for n in ("a", "b", "c")}
        self.assertGreater(len(vals), 1, "proximity collapsed to a class")

    def test_alpha_changes_decay_not_membership(self):
        """Higher alpha decays faster, but nobody drops out of the measure."""
        near = self.L.proximity(self.g, ["origin"], ["c"], alpha=0.6)
        far = self.L.proximity(self.g, ["origin"], ["c"], alpha=0.05)
        self.assertGreater(far, near)
        self.assertGreater(near, 0.0)

    def test_solve_converges_across_tolerances(self):
        """Regression: the previous push implementation moved non-monotonically
        across tolerances and disagreed with a tighter run by up to 100%."""
        vals = list(self.L.check_convergence(self.g, ["c"]).values())
        for v in vals[1:]:
            self.assertAlmostEqual(v, vals[0], places=6)

    def test_repeated_collaboration_strengthens_edge(self):
        p1 = [{"authors": [{"name": "x"}, {"name": "y"}]}]
        p3 = p1 * 3
        self.assertGreater(
            self.L.build_coauthor_graph(p3)["x"]["y"],
            self.L.build_coauthor_graph(p1)["x"]["y"],
        )

    def test_fractional_weighting_damps_consortium_papers(self):
        """A 50-author paper must not outweigh a genuine pairwise collaboration."""
        big = [{"authors": [{"name": f"n{i}"} for i in range(50)]}]
        frac = self.L.build_coauthor_graph(big, fractional=True)
        raw = self.L.build_coauthor_graph(big, fractional=False)
        self.assertAlmostEqual(frac["n0"]["n1"], 1 / 49)
        self.assertAlmostEqual(raw["n0"]["n1"], 1.0)
        self.assertLess(sum(frac["n0"].values()), sum(raw["n0"].values()))

    def test_proximity_is_a_share_in_unit_interval(self):
        p = self.L.proximity(self.g, ["origin"], ["a", "b"])
        self.assertGreater(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_solution_can_be_reused_across_seed_sets(self):
        """One solve serves the observed value and every null draw."""
        sol = self.L.absorption(self.g, ["c"])
        direct = self.L.proximity(self.g, ["origin"], ["c"])
        reused = self.L.proximity(self.g, ["origin"], ["c"], solution=sol)
        self.assertAlmostEqual(direct, reused, places=12)

    def test_unreachable_seed_returns_empty_not_error(self):
        self.assertEqual(self.L.absorption(self.g, ["nobody"]), {})
        self.assertEqual(self.L.proximity(self.g, ["nobody"], ["a"]), 0.0)

    def test_null_ratio_is_near_one_for_unrelated_adopters(self):
        """The null must not credit lineage where there is none."""
        import random
        rng = random.Random(0)
        papers = [
            {"authors": [{"name": f"p{rng.randrange(60)}"},
                         {"name": f"p{rng.randrange(60)}"}]}
            for _ in range(400)
        ]
        g = self.L.build_coauthor_graph(papers)
        nodes = sorted(g)
        res = self.L.proximity_vs_null(
            g, nodes[:5], nodes[30:40], alpha=0.15, n_null=25, seed=1
        )
        self.assertGreater(res["ratio"], 0.3)
        self.assertLess(res["ratio"], 3.0)


class TestCorpusBaselineQuantiles(unittest.TestCase):
    def test_quantile_reports_shape_without_a_cut_point(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "cb", Path(__file__).resolve().parents[1] / "scripts" / "corpus_baselines.py"
        )
        cb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cb)
        vals = list(range(1, 101))
        self.assertEqual(cb.quantile(vals, 0.50), 50)
        self.assertEqual(cb.quantile(vals, 0.99), 99)
        self.assertEqual(cb.quantile([], 0.5), 0.0)
        self.assertEqual(cb.quantile([7], 0.9), 7)


class TestTechniqueRegistry(unittest.TestCase):
    def setUp(self):
        from tacit.registry import load_registry, match_paper, normalise_title
        self.reg = load_registry()
        self.match = match_paper
        self.norm = normalise_title

    def test_registry_ids_unique(self):
        ids = [t.id for t in self.reg]
        self.assertEqual(len(ids), len(set(ids)))

    def test_short_aliases_match_words_not_substrings(self):
        """' mpc ' must not fire on 'compensation'; padding is what makes that work."""
        hit = self.match({"title": "An MPC approach to legged locomotion"}, self.reg)
        self.assertIn("mpc", hit)
        miss = self.match({"title": "Thermal compression and compensation"}, self.reg)
        self.assertNotIn("mpc", miss)

    def test_title_at_either_end_still_matches(self):
        self.assertIn("mpc", self.match({"title": "MPC"}, self.reg))

    def test_multiple_techniques_are_all_kept(self):
        """Techniques co-occur; assigning a paper to one arbitrarily is a cut point."""
        hits = self.match(
            {"title": "Reinforcement learning for impedance control of a soft robot"},
            self.reg,
        )
        self.assertIn("reinforcement_learning", hits)
        self.assertIn("impedance_control", hits)
        self.assertIn("soft_actuator", hits)

    def test_missing_or_empty_title_is_safe(self):
        self.assertEqual(self.match({"title": None}, self.reg), [])
        self.assertEqual(self.match({}, self.reg), [])

    def test_case_and_whitespace_insensitive(self):
        a = self.match({"title": "VISUAL   SERVOING of a manipulator"}, self.reg)
        self.assertIn("visual_servoing", a)


class TestReferenceClassifier(unittest.TestCase):
    def setUp(self):
        from tacit.refs import ReferenceClassifier
        self.c = ReferenceClassifier()

    def test_ieee_stem_strips_instance_ids(self):
        """ICRA, ICRA40945 and ICRA48506 are one venue, not three."""
        for doi in ("10.1109/ICRA.2015.7139278",
                    "10.1109/ICRA40945.2020.9196733",
                    "10.1109/icra48506.2021.9561715"):
            self.assertEqual(self.c.ieee_stem(doi), "ICRA")

    def test_non_ieee_doi_has_no_stem(self):
        self.assertIsNone(self.c.ieee_stem("10.1177/0278364913495721"))

    def test_robotics_and_external_by_doi(self):
        self.assertEqual(self.c.classify({"doi": "10.1109/IROS.2011.6048"})[0], "robotics")
        self.assertEqual(self.c.classify({"doi": "10.1109/CVPR.2016.90"}),
                         ("external", "computer_vision_ml"))
        self.assertEqual(self.c.classify({"doi": "10.1109/TAC.2009.12345"}),
                         ("external", "control_theory"))

    def test_rss_and_ijrr_are_robotics(self):
        self.assertEqual(self.c.classify({"doi": "10.15607/RSS.2018.XIV.001"})[0], "robotics")
        self.assertEqual(self.c.classify({"doi": "10.1177/0278364913495721"})[0], "robotics")

    def test_unknown_is_not_defaulted_to_external(self):
        """Defaulting would inflate B2 reach where metadata is thinnest."""
        self.assertEqual(self.c.classify({"doi": "10.1016/j.example.2019.01.001"}),
                         ("unknown", None))
        self.assertEqual(self.c.classify({}), ("unknown", None))

    def test_venue_text_used_only_as_fallback(self):
        self.assertEqual(
            self.c.classify({"venue": "Proc IEEE Int Conf Robotics Automation"})[0],
            "robotics")

    def test_robotics_venue_wins_over_general_science_substring(self):
        """'Science' is a substring of many venue names; robotics is checked first."""
        self.assertEqual(
            self.c.classify({"venue": "Robotics: Science and Systems"})[0], "robotics")

    def test_summarise_reports_denominator_with_the_ratio(self):
        from tacit.refs import summarise
        refs = [{"doi": "10.1109/IROS.2011.1", "year": "2005"},
                {"doi": "10.1109/CVPR.2016.1", "year": "2014"},
                {"doi": "10.1016/unknown", "year": "2010"}]
        out = summarise(refs, 2016, self.c)
        self.assertAlmostEqual(out["reach"], 0.5)
        self.assertAlmostEqual(out["classified_share"], 2 / 3)
        self.assertEqual(out["n_unknown"], 1)
        self.assertEqual(out["age_mean"], (11 + 2 + 6) / 3)

    def test_summarise_reach_is_none_when_nothing_classified(self):
        from tacit.refs import summarise
        out = summarise([{"doi": "10.1016/x"}], 2020, self.c)
        self.assertIsNone(out["reach"])
        self.assertEqual(out["classified_share"], 0.0)

    def test_implausible_reference_ages_are_dropped(self):
        from tacit.refs import summarise
        out = summarise([{"doi": "10.1109/IROS.2011.1", "year": "2099"}], 2016, self.c)
        self.assertEqual(out["n_aged"], 0)


class TestVenueClassifierPrecision(unittest.TestCase):
    """Regression: a loose 'robotics and automation' match admitted 483 papers
    (2.5% of ICRA) from five other conferences. All strings are real containers
    from the 2006-2025 harvest."""

    def setUp(self):
        from tacit.crossref import classify_container
        self.c = classify_container

    def test_genuine_icra_containers_kept(self):
        for s in [
            "2024 IEEE International Conference on Robotics and Automation (ICRA)",
            "2011 IEEE International Conference on Robotics and Automation",
            "2019 International Conference on Robotics and Automation (ICRA)",
            "Proceedings 2006 IEEE International Conference on Robotics and Automation, 2006. ICRA 2006.",
        ]:
            self.assertEqual(self.c(s), "ICRA", s)

    def test_confusable_conferences_rejected(self):
        for s in [
            "2023 International Conference on Robotics and Automation in Industry (ICRAI)",
            "2016 International Conference on Robotics and Automation for Humanitarian Applications (RAHA)",
            "2018 IEEE 2nd Colombian Conference on Robotics and Automation (CCRA)",
            "2025 International Conference on Intelligent Manufacturing, Robotics and Automation (IMRA)",
            "2024 IEEE International Conference on Advanced Information, Mechanical Engineering, Robotics and Automation (AIMERA)",
        ]:
            self.assertIsNone(self.c(s), s)

    def test_serials_rejected(self):
        for s in ["IEEE Robotics and Automation Letters",
                  "IEEE Transactions on Robotics and Automation",
                  "IEEE Robotics & Automation Magazine"]:
            self.assertIsNone(self.c(s), s)

    def test_iros_containers_kept_including_early_workshop_form(self):
        for s in ["2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)",
                  "2008 IEEE/RSJ International Conference on Intelligent Robots and Systems",
                  "IEEE/RSJ International Workshop on Intelligent Robots and Systems"]:
            self.assertEqual(self.c(s), "IROS", s)
