# flask,scikit-learn,pandas,pickle-mixin

from sys import breakpointhook
from click.core import batch
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app=Flask(__name__)
data = pd.read_csv('/users/rishabhtiwari/documents/GitHub/Bangalore-housing/cleaned_bangalore_data.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():

    locations=  sorted(data['location'].unique())
    return render_template('index.html',locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,float(bhk),float(bath),sqft)
    input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
    prediction = pipe.predict(input)[0] * 1e5



    return str(np.round(prediction,2))

if __name__== "__main__": 
    app.run(debug=True,port=5001)