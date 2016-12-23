import warnings
warnings.filterwarnings("ignore")
import time
import ABSAWeb

from flask import Flask,abort,jsonify,request,redirect,render_template,url_for
from werkzeug.utils import secure_filename
import json
ALLOWED_EXTENSIONS = set(['txt'])


app = Flask(__name__)

@app.route('/')
def index():
  
    return render_template('index.html',
                           title='Home',
                           text = '',time_exe = '')

                           
@app.route('/statistics')
def restaurant_statistics():
   
    return render_template('stats.html',
                           title='Statistics',
                           text = '',time_exe = '')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
        # check if the post request has the file part
        if 'file' not in request.files:
            print 'No file part'
            return redirect('/statistics')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print 'No selected file'
            return redirect('/statistics')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            f = request.files['file']
            text = f.read()
            start = time.time()
            category,aspect_pol = ABSAWeb.bulk_reviews(text)
            end = time.time()
            if len(category.index) == 0:
                  res = "no"
            else:
                  res = "yes"
   
            return render_template('stats.html',
                           title='Statistics',
                           category=category,aspect_pol = aspect_pol,res = res,time_exe = end-start)
@app.route('/getsentiment',methods = ['POST'])
def predict_sentiment():
    text = request.form['text']
    start = time.time()
    ans,summary = ABSAWeb.predict_category_review(text)     
    end = time.time()
    if len(ans.index) == 0:
        res = "no"
    else:
        res = "yes"
    return render_template('index.html',
                           title='Result',
                           ans = ans,summary = summary,res = res,text = text,time_exe = end-start)
                           
#if __name__ == '__main__':
def runMain(vec_fname,lex_filename):
	print 'Initializing models...'
	ABSAWeb.load_models(vec_fname,lex_filename)
	app.run(port = 9000,debug = False,use_reloader=False)