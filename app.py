import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok

# Load the trained model
with open("train_speed_model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
run_with_ngrok(app)  # This makes Flask run in Google Colab

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input to DataFrame
        input_data = pd.DataFrame([{
            "Current_Speed": data["current_speed"],
            "Track_Condition": data["track_condition"],
            "Weather": data["weather"],
            "Obstacle_Detected": data["obstacle_detected"],
            "Station_Proximity": data["station_proximity"]
        }])

        # Predict speed
        predicted_speed = model.predict(input_data)[0]

        # Return JSON response
        return jsonify({"recommended_speed": round(predicted_speed, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run()
    
