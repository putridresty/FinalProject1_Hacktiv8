import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__, template_folder="templates")
model = pickle.load(open("model/model_lr.pkl", "rb"))

@app.route("/")
def main():
    return render_template('index.html')


# Decision Tree
@app.route('/predict', methods=['POST'])
def predict():
   float_features = [float(x) for x in request.form.values()]
   feature = [np.array(float_features)]
   prediction = model.predict(feature)
   
   output = round(prediction[0], 2)

   return render_template('index.html', prediction_text ="$ {}".format(output))

if __name__=="__main__":
    app.run(debug=True)