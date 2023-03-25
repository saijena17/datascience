import numpy as np
import pandas as pd
import pickle
from flask import Flask, request,jsonify, render_template


app = Flask(__name__)
filename = '/Users/saijena/Desktop/datascience/ML/Tbilisi_Housing_Challenge/model.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route("/")
def home():
    return(render_template('index.html'))

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print("Estimated Price: " + str(model.predict([27.0, 2.0, 3.0, 1.0, 50, 30])))
    answer = "Estimated Price: " + str(model.predict([27.0, 2.0, 3.0, 1.0, 50, 30]))
    return(render_template('index.html', prediction_text = answer))


if __name__ == "__main__":
    app.run(debug=True)
