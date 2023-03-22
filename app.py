import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open("model.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features=[np.array(float_feature)]
    prediction=model.predict(features)

    if int(prediction)==1:
        return render_template("index.html",prediction_text="You have {}".format("Heart Disease."))
    else:
        return render_template("index.html", prediction_text="You don't have {}".format("Heart Disease."))


if __name__=="__main__":
    app.run(debug=True)
