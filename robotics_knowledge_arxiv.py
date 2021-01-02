#!/usr/bin/env python
# coding: utf-8

# # Lambda vs Omega Knowledge in Robotics
# This notebook seeks to use topic modeling over a corpus of abstracts robotics paper abstracs to explore a potential relationship between theoretical and practical knowledge in the robotics field and scientific communities generally. Current implementation uses arxiv documents but I hope to use IROS and ICRA abstracts for a more representative picture of the field.
# 
# I took lots of code from https://www.kaggle.com/aiswaryaramachandran/exploring-the-growth-in-ai-using-arxiv/data?select=arxiv-metadata-oai-snapshot.json
# and from https://medium.com/@kurtsenol21/topic-modeling-lda-mallet-implementation-in-python-part-1-c493a5297ad2

# In[1]:


import pandas as pd 
import numpy as np 
from datetime import datetime
import sys
import ast

import plotly.express as px

import nltk
from nltk.corpus import stopwords
import spacy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity


# import networkx
# from networkx.algorithms.components.connected import connected_components

import json
import dask.bag as db



# In[2]:


import utils


# # Extracting Robotics Papers from Arxiv Repo

# In[64]:


def get_arxiv_papers():
    ai_category_list=['cs.RO']
    records=db.read_text("/home/zach/Downloads/*.json").map(lambda x:json.loads(x))
    ai_docs = (records.filter(lambda x:any(ele in x['categories'] for ele in ai_category_list)==True))
    get_metadata = lambda x: {'id': x['id'],
                      'title': x['title'],
                      'category':x['categories'],
                      'abstract':x['abstract'],
                     'version':x['versions'][-1]['created'],
                             'doi':x["doi"],
                             'authors_parsed':x['authors_parsed']}

    data=ai_docs.map(get_metadata).to_dataframe().compute()
    data
    data.to_excel("AI_ML_ArXiv_Papers.xlsx",index=False,encoding="utf-8")
    return data




data = get_arxiv_papers()
data.head()


# The data contains the id, the title,the category the paper belongs to, the date when the version was created and list of authors 

# # Data Preprocessing
# 
# Some preprocessing steps that we need to perform are:
# 
# 1. Extract the Date Time information from version column
# 
# 
# 2. The authors parsed information, first and last names need to be concatenated to get one name.
# 
# 3. Handling Missing DOI's
# 
# 4. We need to look for any possible duplication in the title names
# 

# ### Extracting the Date Time Information

# In[ ]:


data['DateTime']=pd.to_datetime(data['version'])
data.head()


# In[ ]:


import datetime
data['Year'] = data['DateTime'].dt.year
data['Date'] = data['DateTime'].dt.date
#data=utils.extractDateFeatures(data,"DateTime")
data.head()


# ### Cleaning the ***authors_parsed*** column
# 
# 
# 1. Concatenating the authors first and last names.

# In[ ]:


data['num_authors']=data['authors_parsed'].apply(lambda x:len(x))


# In[ ]:


data['authors']=data['authors_parsed'].apply(lambda authors:[(" ".join(author)).strip() for author in authors])
data.head()


# ### Missing DOI 
# 
# In the Data, we can see that there are papers with no doi - Since Arxiv is a pre-print server, once the paper is published DOI is received. This DOI needs to be updated to Arxiv. In cases where there are no DOI - probably they were not published in any other journal or the author forgot to update the doi - hence there is no DOI available
# 
# (Reference : https://academia.stackexchange.com/questions/62480/why-does-arxiv-org-not-assign-dois)

# In[ ]:


print("Number of Papers with No DOI ",data[pd.isnull(data['doi'])].shape[0])


# Aroung 88% of the papers have no DOI - the authors most probably didnt update this information. 

# # Analysing the Data
# 
# 1. How has the field of ML/AI grown over the years?
# 
# 2. Who have been the most successful Authors?
# 
# 3. What are the different topics being spoken about  - and how this has changed over the years?
# 
# 4. Can we cluster papers based on their Abstract and Title? 

# ## Growth in Field of ML AI 

# In[ ]:



papers_over_years=data.groupby(['Year']).size().reset_index().rename(columns={0:'Number Of Papers Published'})
px.line(x="Year",y="Number Of Papers Published",data_frame=papers_over_years,title="Growth of Robotics over the Years")


# From 2010, there has been an exponential growth in this field - and this is continuously increasing over the period of time

# In[ ]:


papers_published_over_days=data.groupby(['Date']).size().reset_index().rename(columns={0:'Papers Published By Date'})
px.line(x="Date",y="Papers Published By Date",data_frame=papers_published_over_days,title="Average Papers Published Over Each Day")


# From one published paper over each day, in the last one year there have been around 100 papers published each day. In Mar2013, there is a jump in number of papers published. Also, 2013 was the year, when the paper on Word2Vec was published - this was a new beginning in the field of NLP

# ## Who has published most papers in AI ML Space

# In[ ]:


ai_authors=pd.DataFrame(data['authors'].tolist()).rename(columns={0:'authors'})
papers_by_authors=ai_authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False).head(20)
px.bar(x="Number of Papers Published",y="authors",data_frame=papers_by_authors.sort_values("Number of Papers Published",ascending=True),title="Top 20 Popular Authors",orientation="h")


# In[ ]:


np.shape(data['authors'].tolist())


# In[ ]:


# import nltk library and then download stopwords
import nltk 
nltk.download('stopwords')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# In[ ]:


abstract_data = list(data.abstract)
abstract_data[:2]


# In[ ]:


stop_words = nltk.corpus.stopwords.words('english')


# In[ ]:


# Build the bigram and trigrams
bigram = gensim.models.Phrases(abstract_data, min_count=20, threshold=100) 
trigram = gensim.models.Phrases(bigram[abstract_data], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[ ]:


# only need tagger, no need for parser and named entity recognizer, for faster implementation
import en_core_web_sm
nlp = en_core_web_sm.load(disable=['parser', 'ner'])


# In[ ]:


def process_words(texts, stop_words=stop_words, allowed_tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
    
    # remove stopwords, short tokens and letter accents 
    texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts]
    
    # bi-gram and tri-gram implementation
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []

    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    
    # remove stopwords and short tokens again after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts_out]    
    
    return texts_out


# In[ ]:


data_ready = process_words(abstract_data)


# In[ ]:


data_ready[0]


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

print('Total Vocabulary Size:', len(id2word))


# In[ ]:


# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]
corpus


# In[ ]:


dict_corpus = {}

for i in range(len(corpus)):
  for idx, freq in corpus[i]:
    if id2word[idx] in dict_corpus:
      dict_corpus[id2word[idx]] += freq
    else:
       dict_corpus[id2word[idx]] = freq


# In[ ]:


dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])


# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(dict_df['freq'], bins=100);


# In[ ]:


dict_df.sort_values('freq', ascending=False).head(10)


# In[ ]:


extension = dict_df[dict_df.freq>1500].index.tolist()
extension


# In[ ]:


stop_words.extend(extension)


# In[ ]:


data_ready = process_words(abstract_data)


# In[ ]:


# Create Dictionary
id2word = corpora.Dictionary(data_ready)

print('Total Vocabulary Size:', len(id2word))


# In[ ]:


# Filter out words that occur less than 10 documents, or more than 50% of the documents.

id2word.filter_extremes(no_below=10, no_above=0.5)

print('Total Vocabulary Size:', len(id2word))


# In[ ]:


# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]


# In[ ]:


mallet_path = '/home/zach/mallet-2.0.8/bin/mallet'


# In[ ]:


ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)


# In[ ]:


from pprint import pprint
# display topics
pprint(ldamallet.show_topics(formatted=False))


# In[ ]:


# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_ready, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('Coherence Score: ', coherence_ldamallet)


# In[ ]:


tm_results = ldamallet[corpus]


# In[ ]:


corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]
corpus_topics


# In[ ]:


topics = [[(term, round(wt, 3)) for term, wt in ldamallet.show_topic(n, topn=20)] for n in range(0, ldamallet.num_topics)]
topics


# In[ ]:


topics_df = pd.DataFrame([[term for term, wt in topic] for topic in topics], columns = ['Term'+str(i) for i in range(1, 21)],
                         index=['Topic '+str(t) for t in range(1, ldamallet.num_topics+1)]).T
topics_df.head()


# In[ ]:


pd.set_option('display.max_colwidth', -1)

topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics], columns = ['Terms per Topic'],
                         index=['Topic'+str(t) for t in range(1, ldamallet.num_topics+1)] )
topics_df


# In[ ]:


import pyLDAvis
import pyLDAvis.gensim
import pyLDAvis.gensim as gensimvis


# In[ ]:


from gensim.models.ldamodel import LdaModel
def convertldaMalletToldaGen(mallet_model):
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha) # original function has 'eta=0' argument
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim


# In[ ]:


ldagensim = convertldaMalletToldaGen(ldamallet)
vis_data = gensimvis.prepare(ldagensim, corpus, id2word, sort_topics=False)
pyLDAvis.display(vis_data)


# In[ ]:


# create a dataframe 
corpus_topic_df = pd.DataFrame()

# get the Titles from the original dataframe
corpus_topic_df['Title'] = data.title

corpus_topic_df['Dominant Topic'] = [item[0]+1 for item in corpus_topics]
corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
corpus_topic_df['Topic Terms'] = [topics_df.iloc[t[0]]['Terms per Topic'] for t in corpus_topics]

corpus_topic_df.head()


# In[ ]:


dominant_topic_df = corpus_topic_df.groupby('Dominant Topic').agg(
                                  Doc_Count = ('Dominant Topic', np.size),
                                  Total_Docs_Perc = ('Dominant Topic', np.size)).reset_index()

dominant_topic_df['Total_Docs_Perc'] = dominant_topic_df['Total_Docs_Perc'].apply(lambda row: round((row*100) / len(corpus), 2))

dominant_topic_df


# In[ ]:


topic_counts = dominant_topic_df[["Dominant Topic", 	"Doc_Count", "Total_Docs_Perc"]]
topic_counts.columns  = ["Dominant Topic", 	"Document Count", "Total Document Percentage"]
topic_counts


# In[ ]:


df_weights = pd.DataFrame.from_records([{v: k for v, k in row} for row in tm_results])
df_weights.columns = ['Topic ' + str(i) for i in range(1,11)]
df_weights


# In[ ]:


df_weights['Year'] = data.Year


# In[ ]:


df_weights.groupby('Year').mean()


# In[ ]:


df_weights['Dominant'] = df_weights.drop('Year', axis=1).idxmax(axis=1)


# In[ ]:


df_weights.groupby('Year')['Dominant'].value_counts(normalize=True)


# In[ ]:


df_dominance = df_weights.groupby('Year')['Dominant'].value_counts(normalize=True).unstack()
df_dominance.reset_index(inplace=True)
df_dominance


# In[ ]:


df_melted = df_dominance.melt(id_vars=['Year'], value_vars=['Topic ' + str(i) for i in range(1,11)], var_name='Topic', value_name='Prevelance')
df_melted


# In[ ]:


sns.relplot(x='Year', y="Prevelance", hue='Topic',
data=df_melted,
kind="line",
height=10,
style="Topic",
dashes=False,
ci=None);


# In[ ]:


# display a progress meter
from tqdm import tqdm

def topic_model_coherence_generator(corpus, texts, dictionary, start_topic_count=2, end_topic_count=10, step=1, cpus=1):
  models = []
  coherence_scores = []
  for topic_nums in tqdm(range(start_topic_count, end_topic_count+1, step)):
    mallet_lda_model = gensim.models.wrappers.LdaMallet(mallet_path=mallet_path, corpus=corpus, num_topics=topic_nums,
                                                            id2word=dictionary, iterations=500, workers=cpus)
      
    cv_coherence_model_mallet_lda = gensim. models.CoherenceModel (model=mallet_lda_model, corpus=corpus, texts=texts,
                                                                     dictionary=dictionary, coherence='c_v')
      
    coherence_score = cv_coherence_model_mallet_lda.get_coherence()
    coherence_scores.append(coherence_score)
    models.append(mallet_lda_model)
  return models, coherence_scores


# In[ ]:


lda_models, coherence_scores = topic_model_coherence_generator(corpus=corpus, texts=data_ready, dictionary=id2word,
                                                               start_topic_count=2, end_topic_count=50, step=2, cpus=16)


# In[ ]:


coherence_df = pd.DataFrame({'Number of Topics': range(2, 51, 2), 'Coherence Score': np.round(coherence_scores, 4)})
coherence_df


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


x_ax = range(2, 51, 2)
y_ax = coherence_scores

plt.figure(figsize=(12, 6))
plt.plot(x_ax, y_ax, c='r')

plt.axhline(y=0.42, c='k', linestyle='--', linewidth=2)
plt.rcParams['figure.facecolor'] = 'white'

xl = plt.xlabel('Number of Topics')
yl = plt.ylabel('Coherence Score')

plt.show()


# In[ ]:





# ## Analyse the Papers published by Bengio Yoshua 
# 
# Bengio Yoshua, is well known for his work on Artifical Neural Networks and Deep Learning. Bengio along with Geoffrey Hinton and Yann LeCun are reffered to as the "Godfathers of AI". Let us look at what kind of research Bengio has been involved him and understand his contributions to this field - that led him to win the Turing Award
# 

# In[ ]:





# In[ ]:


data['is_bengio_author']=data['authors'].apply(lambda x:1 if "Bengio Yoshua" in x else 0)
bengio_papers=data[data['is_bengio_author']==1]
bengio_papers=bengio_papers.reset_index(drop=True)

print("Number of Papers by Bengio Yoshua on Arxiv is ",bengio_papers.shape[0])


# In[ ]:


print("Bengio Yoshua Published His First Paper in ",min(bengio_papers['Date']))
print("Bengio Yoshua Published His Recent Paper in ",max(bengio_papers['Date']))


# Though Bengio, had entered the field of AI ML in the 1990's the first paper published by him on Arxiv is in September of 2010 and his most recent paper is in August 2020. In 10 years, he has published 311 papers - Astounding Rate of Publication. It may be possible that his other papers are tagged into other categories on Arxiv that we are not considering for this analysis

# In[ ]:


bengio_papers_by_year=bengio_papers.groupby(['Year']).size().reset_index().rename(columns={0:'Number of Papers Published'})

px.bar(x="Year",y="Number of Papers Published",title="Papers by Bengio Yoshua Over Years",data_frame=bengio_papers_by_year)


# In[ ]:


print("Average Papers Published in a Year By Bengio Yoshua ",np.median(bengio_papers_by_year['Number of Papers Published']))


# 
# ###  What are the topics in which Bengio Yoshua has published papers in?
# 
# To look at topics at a broad level, we can Build a Frequency Bar Plot to understand key words used in the titles of the papers published.
# 
# Before we look at the top words in the Title, we will have to do some cleaning of the title - Removing Stop Words, Lower Casing the Words. Let us not do any stemming or lemmatization

# In[ ]:


titles=data['title'].tolist()
stop_words = set(stopwords.words('english')) 
titles=[title.lower() for title in titles] ### Lower Casing the Title
titles=[utils.removeStopWords(title,stop_words) for title in titles]


# In[ ]:



bigrams_list=[" ".join(utils.generateNGram(title,2)) for title in titles]
topn=50
top_bigrams=utils.getMostCommon(bigrams_list,topn=topn)
top_bigrams_df=pd.DataFrame()
top_bigrams_df['words']=[val[0] for val in top_bigrams]
top_bigrams_df['Frequency']=[val[1] for val in top_bigrams]
px.bar(data_frame=top_bigrams_df.sort_values("Frequency",ascending=True),x="Frequency",y="words",orientation="h",title="Top "+str(topn)+" Bigrams in Papers by Bengio Yoshua")


# In[ ]:


trigrams_list=[" ".join(utils.generateNGram(title.replace(":",""),3)) for title in titles]
topn=50
top_trigrams=utils.getMostCommon(trigrams_list,topn=topn)
top_trigrams_df=pd.DataFrame()
top_trigrams_df['words']=[val[0] for val in top_trigrams]
top_trigrams_df['Frequency']=[val[1] for val in top_trigrams]
top_trigrams_df=top_trigrams_df[top_trigrams_df["words"]!=""]
px.bar(data_frame=top_trigrams_df.sort_values("Frequency",ascending=True),x="Frequency",y="words",orientation="h",title="Top "+str(topn)+" Trigrams in Papers by Bengio Yoshua")


# As we can there has been a lot of Papers on Recurrent Neural Networks and Reinforcement Learning by Bengio Yoshua. Also, his research areas are also focussed on Neural Machine Translations and Understanding Stochastic Gradients. The top words also, show us that Bengio has worked on various topics in Deep Learning as part of his research - The next question arises is can we categorise his work? And also can we see who are the authors he works predominantly with for each of the categories we have identified

# ### Topic Modelling to Understand Different Themes
# 

# In[ ]:


from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim 


# In[ ]:


'''
The tokenise function will lowercase, and tokenise the sentences
'''

def tokenise(sentences):
    return [gensim.utils.simple_preprocess(sentence, deacc=True,max_len=50) for sentence in sentences]


# In[ ]:


tokenised_sentences=tokenise(bengio_papers['title'].tolist())
tokenised_sentences[0]


# In[ ]:


nlp = spacy.load('en')


# In[ ]:


def lemmatise(sentence,stop_words,allowed_postags=None):
    doc=nlp(sentence)
    #print(sentence)
    if allowed_postags!=None:
        tokens = [token.lemma_ for token in doc if (token.pos_ in allowed_postags) and (token.text not in stop_words)]
    if allowed_postags==None:
        tokens= [token.lemma_ for token in doc if (token.text not in stop_words)]
    return tokens


# In[ ]:


stop_words = spacy.lang.en.stop_words.STOP_WORDS


# In[ ]:


sentences=[" ".join(tokenised_sentence) for tokenised_sentence in tokenised_sentences]
lemmatised_sentences=[lemmatise(sentence,stop_words) for sentence in sentences]
lemmatised_sentences[0]


# #### Building Bigrams and Trigrams 

# In[ ]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(lemmatised_sentences,min_count=2) 
trigram = gensim.models.Phrases(bigram[lemmatised_sentences],min_count=2)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


# In[ ]:


bigrams_words=[bigram_mod[sentence] for sentence in lemmatised_sentences]

trigrams_words=[trigram_mod[sentence] for sentence in bigrams_words]


# #### Creating Dictionary and Corpus 

# In[ ]:


id2word = corpora.Dictionary(trigrams_words)
corpus = [id2word.doc2bow(text) for text in trigrams_words]
[(id2word[id], freq) for id, freq in corpus[0]] 


# In[ ]:


def compute_coherence_values(id2word, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=20,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[ ]:


models,coherence=compute_coherence_values(id2word,corpus,trigrams_words,limit=20,start=2,step=2)
x = range(2, 20, 2)
plt.plot(x, coherence)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# Around 6 topics seem a good number

# In[ ]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=20,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# In[ ]:


pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# In[ ]:


#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#vis


# In[ ]:


print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=trigrams_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# There are topics related to GAN and adversial networks in relation to speech and images.There are topics related to hypergraph and Deep Reinforcement Learning as well.
# 
# Let us now assign, each document to a Topics - a document may consists of more than one topic, but we will assign it the dominant topic

# In[ ]:


def format_topics_sentences(texts,ldamodel=lda_model, corpus=corpus):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)



# In[ ]:


df_topic_sents_keywords = format_topics_sentences(bengio_papers['title'].tolist(),ldamodel=lda_model, corpus=corpus)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)


# In[ ]:


topic_counts=df_dominant_topic['Dominant_Topic'].value_counts().reset_index().rename(columns={'index':'Topic','Dominant_Topic':'Number of Documents'})
topic_counts['percentage_contribution']=(topic_counts['Number of Documents']/topic_counts['Number of Documents'].sum())*100
topic_counts


# We can see that number of documents is each topic is almost equally distributed.. Let us use T-SNE to visualise the topics vs document distribution
# 

# In[ ]:


# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE


# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values


# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)


# In[ ]:


sent_topics_df=pd.DataFrame()
sent_topics_df['Text']=bengio_papers['title'].tolist()
sent_topics_df['tsne_x']=tsne_lda[:,0]
sent_topics_df['tsne_y']=tsne_lda[:,1]
sent_topics_df['Topic_No']=topic_num
sent_topics_df=pd.merge(sent_topics_df,df_dominant_topic,on="Text")
sent_topics_df.head()


# In[ ]:


px.scatter(x='tsne_x',y='tsne_y',data_frame=sent_topics_df,color="Topic_No",hover_data=["Topic_Perc_Contrib"])


# The topics are very well seperated as we can see from TSNE

# ### Has Bengio Worked with Different Authors on Different Topics? Who is the most popular Co-Author across different topics?

# In[ ]:


bengio_papers=pd.merge(bengio_papers,df_dominant_topic.rename(columns={'Text':'title'}),on='title')

num_topics=bengio_papers['Dominant_Topic'].nunique()
authors_df_list=[]

for topic_no in range(num_topics):
    

    temp=bengio_papers[bengio_papers['Dominant_Topic']==topic_no]
    authors=pd.DataFrame(utils.flattenList(temp['authors'].tolist())).rename(columns={0:'authors'})
    authors=authors[authors['authors']!="Bengio Yoshua"]
    papers_authors=authors.groupby(['authors']).size().reset_index().rename(columns={0:'Number of Papers Published'}).sort_values("Number of Papers Published",ascending=False).head(10)
    papers_authors['Topic No']=topic_no
    authors_df_list.append(papers_authors)

co_occurring_authors=pd.concat(authors_df_list)


# In[ ]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[ ]:


fig = make_subplots(rows=3, cols=2)
row=1
col=1
for topic_no in range(num_topics):
    
    wp = lda_model.show_topic(topic_no)
    topic_keywords = ", ".join([word for word, prop in wp])
    temp=co_occurring_authors.loc[co_occurring_authors['Topic No']==topic_no].sort_values("Number of Papers Published",ascending=True)

    fig.add_trace(
    go.Bar(
        x=temp['Number of Papers Published'],
        y=temp['authors'],
        orientation='h',
        name="Topic "+str(topic_no)
        #mode="markers+text",
        #text=["Text A", "Text B", "Text C"],
        #textposition="bottom center"
    ),
    row=row, col=col)
    if col%2==0:
        row=row+1
        col=1
    else:
        col=col+1
fig.update_layout(height=1000, width=1200, title_text="Top 10 Authors With Whom Bengio Worked Across Different Topics")

fig.show()


# Across Topics, Bengio has published papers with Courville Aaron. The other authors are quite distinct across Topics.Courville Aaron is a part of LISA lab along with Bengiom. Except for Topic 2 which talks about self taught deep neural networks and causal networks, the top author with whom Bengio has published his papers with is Courville Aaraon

# # Conclusion and Future Works
# 
# In this Analysis, we started off with analysing the set of AI and ML Papers in the Arxiv Repository. And then we explored the Worked of Bengio Yoshua. As a part of Future Work we can 
# 
# 1. Use Abstract Information for more indepth topic Analysis
# 2. Can we build a co-citation network and analyse similar authors
# 3. We can build a Topic Model on Entire Dataset to understand how each topic has evolved over time

# In[ ]:




