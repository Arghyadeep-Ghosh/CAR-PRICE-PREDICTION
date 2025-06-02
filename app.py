from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Fuel_Type = request.form['Fuel_Type']  # Petrol / Diesel / CNG
        Seller_Type = request.form['Seller_Type']  # Individual / Dealer
        Transmission = request.form['Transmission']  # Mannual / Automatic

        # Process input
        no_year = 2025 - Year
        Kms_Driven_log = np.log(Kms_Driven)

        # One-hot encoding
        Fuel_Type_Diesel = 1 if Fuel_Type == 'Diesel' else 0
        Fuel_Type_Petrol = 1 if Fuel_Type == 'Petrol' else 0
        Seller_Type_Individual = 1 if Seller_Type == 'Individual' else 0
        Transmission_Manual = 1 if Transmission == 'Mannual' else 0

        # Use DataFrame with correct feature names
        input_df = pd.DataFrame([{
            'Present_Price': Present_Price,
            'Kms_Driven': Kms_Driven_log,
            'Owner': Owner,
            'no_year': no_year,
            'Fuel_Type_Diesel': Fuel_Type_Diesel,
            'Fuel_Type_Petrol': Fuel_Type_Petrol,
            'Seller_Type_Individual': Seller_Type_Individual,
            'Transmission_Manual': Transmission_Manual
        }])

        # Make prediction
        prediction = model.predict(input_df)
        output = round(prediction[0], 2)

        # Return prediction result
        if output < 0:
            return render_template('index.html', prediction_text="Sorry, you cannot sell this car.")
        else:
            return render_template('index.html', prediction_text=f"You can sell the car at ₹{output}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)



