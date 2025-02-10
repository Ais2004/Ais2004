import pickle
import pandas as pd

# Load the trained model
with open("train_speed_model.pkl", "rb") as file:
    model = pickle.load(file)

# Function to predict train speed
def predict_speed(current_speed, track_condition, weather, obstacle_detected, station_proximity):
    # Create a dataframe with input values
    input_data = pd.DataFrame([{
        "Current_Speed": current_speed,
        "Track_Condition": track_condition,
        "Weather": weather,
        "Obstacle_Detected": obstacle_detected,
        "Station_Proximity": station_proximity
    }])

    # Predict the speed
    predicted_speed = model.predict(input_data)[0]
    return predicted_speed

# Example Usage
if __name__ == "__main__":
    speed = predict_speed(80, "Moderate", "Clear", 0, 2.5)
    print(f"Predicted Recommended Speed: {speed:.2f} km/h")
