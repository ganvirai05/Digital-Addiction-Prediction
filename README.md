Digital Addiction Prediction System

Overview

This project predicts a user's digital addiction risk level (Low, Moderate, High) based on daily behavioral patterns such as screen time, sleep, social interaction, and phone usage.

It is an end-to-end Machine Learning project that includes:

- Data collection (Google Form)
- Data preprocessing and feature engineering
- Model training and evaluation
- Web application deployment using Flask

---

Problem Statement

With increasing digital usage, it is important to identify addiction patterns early.
This project aims to analyze user behavior and predict addiction risk using Machine Learning techniques.

---

Features

- Predicts addiction level in real-time
- User-friendly web interface
- Handles imbalanced data using SMOTE
- Custom rule-based target engineering
- Trained using Random Forest algorithm
- Clean and interactive UI

---

Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Flask (Web Framework)
- HTML, CSS, Bootstrap

---

Project Workflow

1. Data Collection
   
   - Data collected using Google Forms (primary dataset)

2. Data Preprocessing
   
   - Removed unnecessary columns
   - Handled missing values
   - Converted categorical data

3. Feature Engineering
   
   - Created a custom target variable (addiction risk) using behavioral rules

4. Data Balancing
   
   - Applied SMOTE to handle class imbalance

5. Model Training
   
   - Random Forest Classifier
   - Hyperparameter tuning

6. Evaluation
   
   - Confusion Matrix
   - Classification Report

7. Deployment
   
   - Built a Flask web application
   - User inputs → Model prediction

---

Project Structure

digital-addiction-project/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── model/
│   ├── model.pkl
│   └── scaler.pkl
│
├── src/
│   └── train_model.py
│
├── data/
│   └── final.csv
│
├── requirements.txt
└── README.md

---

How to Run the Project

1. Clone the repository
   git clone <your-repo-link>

2. Install dependencies
   pip install -r requirements.txt

3. Train the model (optional)
   cd src
   python train_model.py

4. Run the web app
   cd app
   python app.py

5. Open browser
   http://127.0.0.1:5000/

---

Sample Input

- Screen Time: 2 hours
- Sleep Duration: 8 hours
- Social Interaction: High

Output

- Low Risk / Moderate Risk / High Risk

---

Key Improvements

- Solved class imbalance using SMOTE
- Created rule-based target variable instead of relying on subjective labels
- Tuned model for better real-world performance

---

Future Enhancements

- Add user login system
- Add dashboard with visual analytics
- Deploy on cloud (live access)
- Mobile-friendly UI

---

Author

Your Name

---

Conclusion

This project demonstrates a complete Machine Learning pipeline from data collection to deployment, solving a real-world problem with practical implementation.