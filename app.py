from flask import Flask, render_template,flash,request,url_for,redirect
import pickle
import re
import warnings
warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings
import nltk                                         #Natural language processing tool-kit
from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF


app =Flask(__name__,template_folder="templates")


@app.route('/')
def first_page():
    return render_template('eg.html')

@app.route('/input')
def  input():
    return render_template("index.html")

@app.route('/text_output')
def text_output():

    text = request.args.get('text')
    org_text=text
    #/assets/header1.css
    svc_model = pickle.load(open("svc_model_new.dat", "rb"))
    tfidf_vectorizer = pickle.load(open("feature_vectorizer_new.pickle", "rb"))

    # PREPROCESSING OF CONVERTED TEXT
    stop_words = ["xxxx", "xxxxxxxx", "the", "when", "get", "told", "got", "one", "they", "even", "asked", "she", "i",
                  "he", "you",
                  'in', 'below', 'them', 'be', 'as', 'into', 'is', 'o', 'few', 'until', 'for', 'his', 'do', 'what',
                  'again', "that'll", 'yourself', 're', 'most', 'y', 'where', 'own', 'about', 'yourselves', 'before',
                  'an', 'been', "she's", 'hers', 'which', 'was', 'did', 'with', 'from', 'themselves', 'ourselves', 'we',
                  'doing', 'should', 'between', 't', 'further', 'him', 'those', 'other', 'your', 'my', 'that', 'd',
                  'there', 'any', 'very', 'only', 'who', 'through', 'up', 'same', 'after', 'the', 'why', 'ours', 'out',
                  'theirs', 'to', 'at', 'her', 'some', 'have', 'here', 'our', 'myself', 'once', "you'll", 's', 'how',
                  'm', 'by', 'such', 'will', 'each', 'while', 'me', 'when', "you'd", 'these', 'it', 'just', 'than',
                  'or', 'having', 'itself', 'too', 'now', 'on', 'himself', 'down', 'so', 'but', "it's", 'its', 'whom',
                  'shan', 'and', 'being', 'herself', 'ma', 'over', 'll', 'are', 've', 'off', 'has', 'both', 'does',
                  'he', 'against', 'then', 'yours', 'all', 'during', 'under', 'this', 'above', 'their', 'am',
                  "should've", 'a', 'more', 'of', 'had', 'were', 'because', "you've", 'they', "you're", 'if', 'can']
    stop = set(stop_words)
    queri = [text]
    temp = []
    snow = nltk.stem.SnowballStemmer('english')
    sentence=text
    sentence = sentence.lower()  # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)  # Removing HTML tags
    sentence = re.sub(r'\d+', r'', sentence)  # Removing numbers
    sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)  # Removing Punctuations

    words = [snow.stem(word) for word in sentence.split() if word not in stop]  # Stemming and removing stopwords
    temp.append(words)

    text = temp
    sent = []
    for row in text:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        sent.append(sequ)
    text = sent


    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    fea = tfidf_vectorizer.transform(queri)
    department = svc_model.predict(fea)


    global dept
    if department == 0:
        dept="DEBT COLLECTION SUPPORT TEAM"
    elif department == 1:
        dept="MORTGAGE SUPPORT TEAM"
    elif department == 2:
        dept="CREDIT REPORTING SUPPORT TEAM"
    elif department == 3:
        dept="CREDIT CARD SUPPORT TEAM"
    elif department == 4:
        dept="BANK ACCOUNT OR SERVICE SUPPORT TEAM"
    elif department == 5:
        dept="STUDENT LOAN SUPPORT TEAM"
    elif department == 6:
        dept="CONSUMER LOAN SUPPORT TEAM"
    elif department == 7:
        dept="PAYDAY LOAN SUPPORT TEAM"
    else:
        dept="MONEY TRANSFERS SUPPORT TEAM"

    return {"dept": dept, "org_text": org_text}





if __name__=='__main__':
	app.run(debug=True)