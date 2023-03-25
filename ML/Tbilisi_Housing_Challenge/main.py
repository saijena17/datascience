import numpy as np
import pandas as pd
import pickle
from flask import Flask, request,jsonify, render_template


app = Flask(__name__)
filename = './model.pkl'
model = pickle.load(open(filename, 'rb'))

#as soon as we hit http://127.0.0.1:5000/ this is displayed
@app.route("/")
def home():
    return(render_template('index.html'))

#as soon as we fill details and hit Predict button, it will call /predict answer is displayed
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #generator
    if request.method == 'POST':
        #fetching a particular column value
        space = request.form['space']
        print(space)
        print(request.form.values())
        #getting values from generator in form of string i.e   i is str here and we can convert it int(i)
        print([i for i in request.form.values()])
        print("Estimated Price: " + str(model.predict([int(i) for i in request.form.values()])))
        answer = "Estimated Price: " + str(model.predict([int(i) for i in request.form.values()]))
    return(render_template('index.html', prediction_text = answer))



if __name__ == "__main__":
    app.run(debug=True)
