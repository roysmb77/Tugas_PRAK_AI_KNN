from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
from sklearn import datasets
import plotly.express as px
import pandas as pd
import os

app = Flask(__name__)

# Load model, scaler, dan data iris
model = joblib.load("models/knn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
iris = datasets.load_iris()
labels = iris.target_names

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    input_data = np.array([[sl, sw, pl, pw]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    predicted_label = labels[prediction]
    probs = {labels[i]: round(probabilities[i] * 100, 2) for i in range(len(labels))}

    # Grafik
    df = pd.DataFrame({'Jenis Bunga': labels, 'Probabilitas (%)': probabilities * 100})
    fig = px.bar(df, x='Jenis Bunga', y='Probabilitas (%)', title='Probabilitas Prediksi KNN', color='Jenis Bunga')
    fig.update_layout(template='plotly_white')
    graph_html = fig.to_html(full_html=False)

    return render_template("result.html", prediction=predicted_label, probabilities=probs, graph_html=graph_html)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)