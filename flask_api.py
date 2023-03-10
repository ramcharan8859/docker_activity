from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("BankNote.pkl","rb")
Banknote=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
   
    """class predictor
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: kurtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The Prediction is
       
    """
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    kurtosis=request.args.get("kurtosis")
    entropy=request.args.get("entropy")
    prediction=clf.predict([[variance,skewness,kurtosis,entropy]])
    print(prediction)
    return "Prediction is "+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """class predictor
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
     
    responses:
        200:
            description: The Prediction is
       
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=clf.predict(df_test)
   
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)