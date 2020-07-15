import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from joblib import load

app = Flask(__name__)
model = pickle.load(open('naive_bayes.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('card.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[float(x) for x in request.form.values()]]
    print(x_test)
    sc=load('detection.save')
    prediction = model.predict(sc.transform(x_test))
    print(prediction)
    output=prediction[0]

    if(output==0):
        pred="Transaction Successful"
    else:
        pred="Fraudulent Credit Card Transaction"
    return render_template('card.html',prediction_text='Status: {}'.format(pred))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
