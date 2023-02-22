from crypt import methods
from importlib.resources import contents
from flask import Flask, render_template, request
import os
import NLTK_alolitm
from selectolax.parser import HTMLParser

app = Flask(__name__, template_folder='template')
temp  = os.getcwd() + "/Document"
name = []

@app.route('/')
def upload_file():
    print(temp)
    return render_template('index.html')

@app.route('/gensim')
def gens():
    return render_template('gensim.html')

@app.route('/show_file', methods = ['POST'])
def display_file():
    f = request.files.getlist('file[]')
    for i in f :
        comm = os.path.join(temp,i.filename)
        i.save(comm)
        name.append(comm)   
        
    return render_template('gensim.html',contents = name) 

@app.route('/search_word',methods = ['POST'])
def search_word():
    key = request.form['key']
    res = NLTK_alolitm.Algolitm(name,key)
    
    return render_template('gensim.html',res = res)

@app.route('/top5',methods = ['POST'])
def top5w():
    BOW_hower = NLTK_alolitm.BOW()
    return render_template('gensim.html',hower = BOW_hower)

@app.route('/top5tf',methods = ['POST'])
def top5if():
    Tfidf = NLTK_alolitm.Tf_idf()
    return render_template('gensim.html',tfidf = Tfidf)

@app.route('/fakenews')
def page_faken():
    return render_template('fakeN.html')

@app.route('/fakenews',methods = ['POST'])
def fake_n():
    result = ""
    fn = NLTK_alolitm.fake_news(request.form['T'])
    if(fn) : result = "fake"
    else : result = "reliable"
    return render_template('fakeN.html',result = result)

@app.route('/spacy')
def page_spacy():
    return render_template('spacy.html')

@app.route('/spacy',methods = ['POST'])
def spacy():
    spacy_text = request.form['spacy_text']
    spacy_label = request.form.getlist('spacy_label')
    display = NLTK_alolitm.pocess_spacy(spacy_text, spacy_label)
    return render_template('spacy.html',display = display,spacy_text = spacy_text)

@app.route('/sent')
def page_sent():
    return render_template('sent.html')

@app.route('/sent',methods = ['POST'])
def sentim():
    sent = NLTK_alolitm.pocess_sent(request.form['senti'])
    return render_template('sent.html',sent = sent)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug = True)

