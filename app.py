import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
modelGB = pickle.load(open('model_digits_GB', 'rb'))
# modelDT = pickle.load(open('model_digits_DT', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    final_features = list(map(int, feature_list))
    
    predictionGB = modelGB.predict(final_features)
   # predictionDT = modelDT.predict(final_features)

    return render_template('index.html', prediction_text='Naive Bayes predicts {}'.format(predictionGB))


if __name__ == "__main__":
    app.run(debug=True)
