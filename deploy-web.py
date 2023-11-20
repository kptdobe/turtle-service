from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
from fastai.vision.all import *
app = Flask(__name__)
CORS(app, support_credentials=True)

def is_cat(x): return x[0].isupper() 

# load the learner
learn = load_learner('./model/model.pkl')
categories = ('Dog', 'Cat')

def classify_images(img):
    i = PILImage.create(img)
    #'Is it a car?', 'Is it a car? but as zero or one', 'probabillity of [dog, cat]'
    pred, idx, probs = learn.predict(i)
    #return dictionary
    #zip together the categories and the
                                        #turn probs to float
    
    return { 
        'prediction': categories[idx],
        'probabilities': dict(zip(categories, map(float, probs)))
    }

# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(classify_images(request.files['image']))

if __name__ == '__main__':
    app.run()