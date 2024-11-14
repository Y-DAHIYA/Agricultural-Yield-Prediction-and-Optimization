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

# Load your dataset to read unique categorical values (assuming you have a dataset)
data = pd.read_csv('C:/Users/JDPK/Downloads/agriculture_dataset.csv')

if 'Farm_ID' in data.columns:
    data = data.drop('Farm_ID', axis=1)

# Identify categorical columns
cat_cols = data.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# Set the title of the app
st.title("AI-Based Agricultural Yield Forecasting and Optimization")

# Page 1: Input Form
if 'page' not in st.session_state:
    st.session_state.page = 'input'

if st.session_state.page == 'input':
    # Create a dictionary to store user inputs
    user_input = {}

    # Loop over each categorical column and create a dynamic selectbox
    for column in cat_cols:
        unique_values = data[column].unique()
        user_input[column] = st.selectbox(f"Select {column}", options=unique_values)

    # Add numerical inputs (e.g., for area, rainfall, etc.)
    user_input['Farm_Area(acres)'] = st.number_input("Enter Farm Area (in acres)", min_value=1, step=1)
    user_input['Fertilizer_Used(tons)'] = st.number_input("Enter Fertilizer Used (in tons)", min_value=0.0, step=0.1)
    user_input['Pesticide_Used(kg)'] = st.number_input("Enter Pesticide Used (in kg)", min_value=0.0, step=0.1)
    user_input['Water_Usage(cubic meters)'] = st.number_input("Enter Water Usage (in cubic meters)", min_value=0, step=1)

    # Predict button
    if st.button("Predict"):
        # Save the user input to session state
        st.session_state.user_input = user_input

        # Transition to prediction page
        st.session_state.page = 'prediction'
        st.rerun()  # Use rerun to refresh the app and show the prediction page

# Page 2: Display Prediction Result
if st.session_state.page == 'prediction':
    if 'user_input' not in st.session_state:
        st.warning("Please complete the inputs first.")
    else:
        # Retrieve user inputs from session state
        user_input = st.session_state.user_input

        # Convert user input into a DataFrame
        user_input_df = pd.DataFrame(user_input, index=[0])

        # Encoding the categorical features based on the input data
        user_input_encoded = pd.get_dummies(user_input_df, columns=cat_cols, drop_first=True)

        # Ensure the columns match the trained model's features by reindexing
        user_input_encoded = user_input_encoded.reindex(columns=encoded_columns, fill_value=0)

        # Ensure the data passed to scaler has the same format (NumPy array without feature names)
        user_input_scaled = scaler.transform(user_input_encoded.values)

        # Define the function to make predictions
        def predict(model, input_data):
            prediction = model.predict(input_data)
            return prediction

        # Make the prediction
        prediction = predict(rf_clf, user_input_scaled)

        # Display the prediction in the main screen
        st.subheader("Predicted Agricultural Yield")
        st.write(f"The predicted yield is: {prediction[0]} tons/hectare")

        # Optionally, a button to go back to input page
        if st.button("Go back to Input Page"):
            st.session_state.page = 'input'
            st.rerun()  # Refresh the app to go back to the input page
