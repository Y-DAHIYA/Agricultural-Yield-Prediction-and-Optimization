import streamlit as st
import pickle
import pandas as pd

# Load saved models and encoders using pickle
with open('random_forest_agri_model.pkl', 'rb') as f:
    rf_clf = pickle.load(f)

with open("columns_order_agri.pkl", "rb") as f:
    encoded_columns = pickle.load(f)

with open('scaler_agri.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load individual encoders
with open('crop_encoder.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)

with open('irrigation_encoder.pkl', 'rb') as f:
    irrigation_encoder = pickle.load(f)

with open('soil_encoder.pkl', 'rb') as f:
    soil_encoder = pickle.load(f)

with open('season_encoder.pkl', 'rb') as f:
    season_encoder = pickle.load(f)

# Set the title of the app
st.title("AI-Based Agricultural Yield Forecasting and Optimization")

# Add sidebar for user input
st.sidebar.header("Input Features")

# Create a dictionary to store user inputs
user_input = {}

# Manually provide the options for the categorical columns based on the encoders
user_input['Crop_Type'] = st.sidebar.selectbox("Select Crop Type", options=crop_encoder.classes_)
user_input['Irrigation_Type'] = st.sidebar.selectbox("Select Irrigation Type", options=irrigation_encoder.classes_)
user_input['Soil_Type'] = st.sidebar.selectbox("Select Soil Type", options=soil_encoder.classes_)
user_input['Season'] = st.sidebar.selectbox("Select Season", options=season_encoder.classes_)

# Add numerical inputs (e.g., for area, rainfall, etc.)
user_input['Farm_Area(acres)'] = st.sidebar.number_input("Enter Farm Area (in acres)", min_value=1, step=1)
user_input['Fertilizer_Used(tons)'] = st.sidebar.number_input("Enter Fertilizer Used (in tons)", min_value=0.0, step=0.1)
user_input['Pesticide_Used(kg)'] = st.sidebar.number_input("Enter Pesticide Used (in kg)", min_value=0.0, step=0.1)
user_input['Water_Usage(cubic meters)'] = st.sidebar.number_input("Enter Water Usage (in cubic meters)", min_value=0, step=1)


# Encode the categorical features using the LabelEncoder
user_input['Crop_Type'] = crop_encoder.transform([user_input['Crop_Type']])[0]
user_input['Irrigation_Type'] = irrigation_encoder.transform([user_input['Irrigation_Type']])[0]
user_input['Soil_Type'] = soil_encoder.transform([user_input['Soil_Type']])[0]
user_input['Season'] = season_encoder.transform([user_input['Season']])[0]

# Create DataFrame for user input
user_input_df = pd.DataFrame([[
    user_input['Crop_Type'], 
    user_input['Irrigation_Type'], 
    user_input['Soil_Type'], 
    user_input['Season'], 
    user_input['Farm_Area(acres)'], 
    user_input['Fertilizer_Used(tons)'], 
    user_input['Pesticide_Used(kg)'], 
    user_input['Water_Usage(cubic meters)']
]], columns=["Crop_Type", "Irrigation_Type", "Soil_Type", "Season", "Farm_Area(acres)", 
            "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Water_Usage(cubic meters)"])


# Ensure the columns match the trained model's features by reindexing
user_input_df = user_input_df.reindex(columns=encoded_columns, fill_value=0)

# If Yield(tons) is not necessary, remove it from the columns
if 'Yield(tons)' in user_input_df.columns:
    user_input_df = user_input_df.drop(columns=['Yield(tons)'])

# Scale numerical features using the scaler
numerical_columns = ["Farm_Area(acres)", "Fertilizer_Used(tons)", "Pesticide_Used(kg)", "Water_Usage(cubic meters)"]
scaled_features = scaler.transform(user_input_df[numerical_columns])
user_input_df[numerical_columns] = scaled_features

# Prediction button
if st.sidebar.button("Predict"):
    # Define the function to make predictions
    def predict(model, input_data):
        prediction = model.predict(input_data)
        return prediction

    # Make the prediction
    prediction = predict(rf_clf, user_input_df)

    # Display the prediction in the main screen
    st.subheader("Predicted Agricultural Yield")
    st.write(f"The predicted yield is: {prediction[0]} tons/hectare")

    # Display user input data
    st.subheader("User Input Data")
    st.write(user_input_df)
