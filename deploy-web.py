from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
from fastai.vision.all import *
import __main__

def is_cat(x): return x[0].isupper()
__main__.is_cat = is_cat

app = Flask(__name__)
CORS(app, support_credentials=True)

# load the learner
learn = load_learner('./model/model.pkl')
categories = ('Dog', 'Cat')

def classify_images(img):
    i = PILImage.create(img)
    pred, idx, probs = learn.predict(i)

    return { 
        'prediction': categories[idx],
        'probabilities': dict(zip(categories, map(float, probs)))
    }

# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(classify_images(request.files['image']))

@app.route('/')
def hello_world():
    return 'Hello, do you want to know where is your turtle?'

app.run()