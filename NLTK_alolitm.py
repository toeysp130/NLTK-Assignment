from cgi import print_directory
import itertools
from logging.config import dictConfig
from typing import List
from textblob import TextBlob
from gensim.models.tfidfmodel import TfidfModel
from collections import defaultdict
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers.file_utils import is_tf_available,is_torch_available,is_torch_tpu_available
from transformers import BertTokenizerFast,BertForSequenceClassification
from transformers import Trainer, TrainingArguments, AutoTokenizer,AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import spacy
from spacy import displacy

articles = []
def Algolitm(name,key) :
    for i in name : 
        f = open(i,"r")
        article = f.read()
        tokens = word_tokenize(article)
        lower_token = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_token if t.isalpha()]
        no_stop = [t for t in alpha_only if t not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stop]
        articles.append(lemmatized)
    
    global dictionary 
    dictionary = Dictionary(articles)
    computer_id = dictionary.token2id.get(key)
    # print(computer_id)
    # print(dictionary.get(computer_id))
    if(computer_id != None) :
        print(dictionary)
        print('%'*30)
        return "Ok! this keyword found!!"
    else :
        return "Sorry keyword not found"

def BOW() :
    print(dictionary)
    top5 = []
    corpus = [dictionary.doc2bow(a) for a in articles]
    total_word_count = defaultdict(int)
    for word_id,word_count in itertools.chain.from_iterable(corpus) : 
        total_word_count[word_id] += word_count
    sorted_word_count = sorted(total_word_count.items(),key = lambda w : w[1], reverse=True)

    for word_id, word_count in sorted_word_count[:5] : 
        t = dictionary.get(word_id),word_count
        top5.append(t)
    return top5

def Tf_idf() :
    tfidf_ = []
    corpus = [dictionary.doc2bow(a) for a in articles]
    doc = corpus[0]
    tfidf = TfidfModel(corpus)
    tfidf_weights = tfidf[doc]
    sorted_tfidf_weight = sorted(tfidf_weights,key=lambda w: w[1] , reverse=True)
    print(tfidf_weights[:5])
    for word_id, word_count in doc[:5] : 
        t = dictionary.get(word_id),word_count
        tfidf_.append(t)
    return tfidf_

def fake_news(text):
    convert_to_label = False
    model_path = "fake-news-model" 
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512,return_tensors="pt")
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    d = {
    0: "reliable",
    1: "fake"
    }
    print(probs)
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())


def pocess_spacy(text, list_check):
    #nlp = spacy.load('/opt/anaconda3/envs/NLPENV/lib/python3.7/site-packages/en_core_web_sm/en_core_web_sm-2.3.1')
    nlp = spacy.load('en_core_web_sm')
    listAll = list(['PERSON', 'NORP', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'work_of_art',
                'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'])
    doc = nlp(text)
    if 'all' in list_check:
        list_check = listAll
    options = {"ents": list_check}
    return displacy.render(doc, style='ent', options=options)



def pocess_sent(Text):
    blob_text = TextBlob(Text)
    level_sentiment = blob_text.sentiment.polarity
    if level_sentiment >= -1.0 and level_sentiment < 0.0:
        return "Negative"
    elif level_sentiment > 0.0:
        return "Positive"
    return "Neutral"
