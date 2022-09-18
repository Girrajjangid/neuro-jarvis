import os
import numpy as np
import pandas as pd
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
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    data=[x for x in request.form.values()]
    print(data)
    text = data[0]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_pickle('processed')
    embedded = pd.read_pickle('embedding')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return jsonify(output)


@app.route('/predict',methods=['POST'])
def predict():
    data=[x for x in request.form.values()]
    print(data)
    text = data[0]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_pickle('processed')
    embedded = pd.read_pickle('embedding')
    new_embeddings1 = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedded, new_embeddings1)
    max_score_index = torch.argmax(cosine_scores).item()
    output = df.iloc[max_score_index].answer 
    print(output)
    return render_template("index.html",prediction_text="{}".format(output))


if __name__ == '__main__':
    app.run(debug=True)
