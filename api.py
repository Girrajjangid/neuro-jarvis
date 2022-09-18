<<<<<<< HEAD
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import torch
from sentence_transformers import SentenceTransformer, util


# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    text = np.array(list(data.values())).reshape(1,-1)[0]
    #text = "Sir can you please help me"
    text = [str(text)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    file_path = os.path.join(basepath, 'dataset/')
    df = pd.read_csv(f'{file_path}processed.csv')
    embedded = torch.load(f'{file_path}embedding.pt')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return jsonify(output)


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    text = np.array(data).reshape(1,-1)[0]
    #text = "Sir can you please help me"
    text = [str(text)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    file_path = os.path.join(basepath, 'dataset/')
    df = pd.read_csv(f'{file_path}processed.csv')
    embedded = torch.load(f'{file_path}embedding.pt')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return render_template("home.html",prediction_text="{}".format(output))


if __name__ == '__main__':
    app.run(debug=True)


=======
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import torch
from sentence_transformers import SentenceTransformer, util


# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    # Main page
    return render_template('home.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    text = np.array(list(data.values())).reshape(1,-1)[0]
    #text = "Sir can you please help me"
    text = [str(text)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    file_path = os.path.join(basepath, 'dataset/')
    df = pd.read_csv(f'{file_path}processed.csv')
    embedded = torch.load(f'{file_path}embedding.pt')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return jsonify(output)


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    text = np.array(data).reshape(1,-1)[0]
    #text = "Sir can you please help me"
    text = [str(text)]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    file_path = os.path.join(basepath, 'dataset/')
    df = pd.read_csv(f'{file_path}processed.csv')
    embedded = torch.load(f'{file_path}embedding.pt')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return render_template("home.html",prediction_text="{}".format(output))


if __name__ == '__main__':
    app.run(debug=True)


>>>>>>> 33b57af31bca84a1a95bbf2c00622c856fa7f254
