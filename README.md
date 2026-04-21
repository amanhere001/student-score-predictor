📊 Student Score Predictor
A machine learning project that predicts a student's exam score based on the number of study hours using Linear Regression.

🚀 Features
Clean dataset with 30 student records
Linear Regression model training
Model evaluation using MAE, RMSE, and R² Score
Actual vs Predicted score comparison table
Custom input prediction for any number of study hours
🛠️ Tech Stack
Tool	Purpose
Python	Core language
Pandas	Data handling
NumPy	Numerical operations
Scikit-learn	ML model & evaluation
Matplotlib	Data visualization
📁 Project Structure
student-score-predictor/
│
├── score_predictor.py   # Main Python script
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
⚙️ How to Run
1. Clone the repository

git clone https://github.com/yourusername/student-score-predictor.git
cd student-score-predictor
2. Install dependencies

pip install -r requirements.txt
3. Run the predictor

python score_predictor.py
📊 Sample Output
=======================================================
     STUDENT SCORE PREDICTOR - Amandeep Kumar
=======================================================

Dataset Overview:
Total students  : 30
Avg Study Hours : 5.72 hrs
Avg Score       : 68.43 marks

--- Model Parameters ---
Equation : Score = 7.83 x Hours + 18.24

--- Model Evaluation ---
Mean Absolute Error (MAE) : 2.41
R² Score                  : 0.9201  (92.01% variance explained)

--- Predict Score for Custom Study Hours ---
Study Hours: 2.0 hrs  -->  Predicted Score: 33.9/100
Study Hours: 5.0 hrs  -->  Predicted Score: 57.4/100
Study Hours: 9.0 hrs  -->  Predicted Score: 88.7/100
📌 Key Concepts Used
Linear Regression: Models the relationship between study hours (input) and exam score (output) as a straight line
Train/Test Split: 80% training data, 20% testing data
R² Score: Measures how well the model explains the variance in scores (0.92 = 92%)
MAE & RMSE: Measure average prediction error in score units
👨‍💻 Author
Amandeep Kumar
B.Tech CSE | Usha Martin University
📧 peednama01@gmail.com
