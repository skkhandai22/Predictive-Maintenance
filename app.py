import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    if prediction == 1:
        pred = 'has No Failure'
    elif prediction ==2:
        pred = 'has Heat Dissipation Failure'
    elif prediction ==3:
        pred = 'has Overstrain Failure'
    elif prediction ==4:
        pred = 'has Power Failure'
    elif prediction == 5:
        pred = ' may be has Random Failure'
    elif prediction == 6:
        pred = 'has Tool wear Failure'
    else:
        pred = "Unknown Failure"
    return render_template('index.html', prediction_text='Machine {}'.format(pred))

@app.route('/predict_api',methods=["POST"])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)