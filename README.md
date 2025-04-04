🌾 Agriculture Yield Prediction Web App

📌 Description
The Agriculture Yield Prediction Web App is a machine learning-powered tool that predicts the expected crop yield per hectare based on key inputs such as crop type, temperature, rainfall, and pesticide usage. Designed using Streamlit, this app allows farmers, researchers, and policymakers to make informed decisions about crop planning and resource allocation.

🚀 Try the App Live
👉 Click Here to Use the App

✨ Features
📊 Predict yield for different crops based on environmental factors

🔍 User-friendly interface for input and instant predictions

📈 Trained using Random Forest Regression

🧠 Option to retrain the model with new data

💾 Supports CSV input for batch predictions

🛠️ Tech Stack
Tool	Purpose
Python	Core Programming
Pandas, NumPy	Data Handling
Scikit-learn	Machine Learning
Streamlit	Web App UI
Matplotlib & Seaborn	Data Visualization
PIL	Image Processing
📷 Screenshots
🎯 Home Page:

🧮 Prediction Page:

📈 Model Metrics:

🗂️ Dataset Info
The app uses a cleaned and preprocessed dataset with the following features:

Crop — Type of crop

Temperature — Average annual temperature (°C)

Rainfall — Annual rainfall (mm)

Pesticide Usage — kg per hectare

Yield — Target variable (kg/hectare)

💻 How to Run Locally
bash
Copy
Edit
# Clone the repo
git clone https://github.com/yourusername/agriculture-yield-app.git
cd agriculture-yield-app

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
📦 Deployment
The app is deployed using Streamlit Cloud. You can deploy yours using:

Push your code to GitHub

Go to Streamlit Community Cloud

Connect your GitHub repo and deploy

🧠 Machine Learning Model
Algorithm: Random Forest Regressor

Evaluation Metric: R² Score

Model Accuracy: ~90% on test data

🙋‍♂️ Author
👨‍💻 Yash
Data Analyst & Python Developer
🔗 LinkedIn | 🌐 Portfolio

📚 Lessons Learned
Learned how to build interactive UI with Streamlit

Understood the full ML pipeline: preprocessing → training → prediction

Gained experience in deploying ML models to the web

Practiced building user-focused features with real-world impact

