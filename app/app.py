from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("../model/model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# Label mappings (must match training)
age_map = {"15-24":0, "25-34":1, "35-44":2, "45+":3}
yes_no_map = {"yes":1, "no":0}
mood_map = {"rarely":0, "sometimes":1, "often":2}

# ---------------- AI SUGGESTION FUNCTION ----------------
def generate_suggestions(data, risk):
    tips = []

    if data["screen_time"] > 6:
        tips.append("Reduce daily screen time to under 5 hours")

    if data["sleep_duration"] < 6:
        tips.append("Try to get at least 7 hours of sleep")

    if data["phone_checks"] > 15:
        tips.append("Limit phone checking frequency using focus mode")

    if data["social_time"] < 30:
        tips.append("Increase offline social interactions")

    if data["mood_swings"] == "often":
        tips.append("Take regular digital detox breaks")

    # Risk-based advice
    if risk == "high":
        tips.append("High risk detected: consider a structured digital detox plan")

    elif risk == "moderate":
        tips.append("Moderate usage: maintain a better balance between online and offline life")

    else:
        tips.append("Great! Keep maintaining healthy digital habits")

    return tips

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    age = request.form["age_group"]
    screen_time = float(request.form["screen_time"])
    social_media = float(request.form["social_media"])
    gaming = float(request.form["gaming"])
    streaming = float(request.form["streaming"])
    phone_checks = float(request.form["phone_checks"])
    sleep_duration = float(request.form["sleep_duration"])
    sleep_delay = request.form["sleep_delay"]
    work_hours = float(request.form["work_hours"])
    irritation = request.form["irritation"]
    mood = request.form["mood_swings"]
    social_time = float(request.form["social_time"])

    # Encode input
    input_data = np.array([[
        age_map[age],
        screen_time,
        social_media,
        gaming,
        streaming,
        phone_checks,
        sleep_duration,
        yes_no_map[sleep_delay],
        work_hours,
        yes_no_map[irritation],
        mood_map[mood],
        social_time
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Decode output
    labels = {0:"Low", 1:"Moderate", 2:"High"}
    result = labels[prediction]

    # Prepare data for suggestions
    user_data = {
        "screen_time": screen_time,
        "sleep_duration": sleep_duration,
        "phone_checks": phone_checks,
        "social_time": social_time,
        "mood_swings": mood
    }

    suggestions = generate_suggestions(user_data, result.lower())

    return render_template("index.html",
                           prediction_text=f"Addiction Risk: {result}",
                           suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
