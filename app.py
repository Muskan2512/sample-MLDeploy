from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(data)])[0]
    return jsonify({"prediction": int(prediction)}) #['setosa' 'versicolor' 'virginica']

if __name__ == '__main__':
    app.run(debug=True)
