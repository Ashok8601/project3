from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open("multi_linear.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def home():
    predicted_price = None
    if request.method == "POST":
        # Get values from form
        area = float(request.form.get("area"))
        bedrooms = int(request.form.get("bedrooms"))
        age = float(request.form.get("age"))

        # DataFrame for prediction
        input_data = pd.DataFrame([[ bedrooms, age,area]], columns=[ "bedrooms", "age","area"])
        predicted_price = model.predict(input_data)[0]
        predicted_price = round(predicted_price, 2)

    return render_template("home.html", predicted_price=predicted_price)

if __name__ == "__main__":
    app.run(debug=True)
