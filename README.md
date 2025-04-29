# Heart Disease Prediction System

This machine learning project predicts the likelihood of heart disease in individuals based on various health metrics. The system uses classifiers like **Decision Tree** and **Random Forest** to make predictions. The app is built using **Streamlit** for real-time, interactive predictions.

## 📈 Introduction

Heart disease is one of the leading causes of death globally. Early detection can help prevent serious health issues. This project aims to predict whether a person is at risk of heart disease based on their health information. By analyzing factors like age, cholesterol levels, blood pressure, and others, the system can classify whether a person has heart disease (1) or not (0).

## 🧠 Concepts Used

- **Machine Learning**: The project uses **Supervised Learning** models to predict the target variable (whether the person has heart disease).
- **Decision Tree Classifier** and **Random Forest Classifier**: These models are trained on the dataset to classify individuals based on their health data.
- **Streamlit**: A Python library used to create an interactive and user-friendly web app.
- **Outlier Handling**: Winsorization technique is used to handle outliers in the data to improve model accuracy.
- **Feature Engineering**: New features are created from existing ones (e.g., `age^2`) to enhance the model's performance.

## ⚙️ How It Works

1. **Data Preprocessing**: 
   - The dataset is first cleaned by filling missing values with the median or mode.
   - Outliers are handled using the Winsorization technique to improve the model’s robustness.
   - Feature engineering is performed to create additional features such as `age^2` to help improve model accuracy.

2. **Model Training**: 
   - Two classifiers, **Decision Tree** and **Random Forest**, are trained on the dataset.
   - The data is split into training and testing sets using **train-test split** from `sklearn`.

3. **Streamlit Web App**: 
   - The app allows users to input their health data such as age, sex, cholesterol, blood pressure, etc.
   - Once the user inputs the data, the model predicts whether the user has heart disease or not and displays the result.
   
4. **Model Evaluation**: 
   - The performance of both models is evaluated based on training and testing accuracy, which is displayed on the web app.

## 🧑‍💻 Technologies Used

- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical operations and handling arrays.
- **Scikit-learn**: For implementing machine learning algorithms (Logistic Regression, Decision Tree, Random Forest) and evaluation metrics (accuracy, confusion matrix).
- **Streamlit**: For creating the interactive web app.
- **Matplotlib, Seaborn**: For visualizing the data (optional for exploratory analysis).
- **Feature-engine**: For outlier handling with Winsorization.

## 🔢 User Inputs

The user can input the following health metrics:
- Age
- Sex (0 = Female, 1 = Male)
- Chest Pain Type (0-3)
- Resting Blood Pressure
- Cholesterol Levels
- Fasting Blood Sugar
- Resting Electrocardiographic Results (0-2)
- Maximum Heart Rate Achieved
- Exercise Induced Angina (1 = Yes, 0 = No)
- ST Depression
- Slope of Peak Exercise (0-2)
- Number of Major Vessels (0-4)
- Thalassemia (0-3)

## ⚙️ How to Run the App

1. **Save the code** as `heart_disease_app.py`.
2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn streamlit feature_engine matplotlib seaborn
   ```
3. **Download the dataset** (or use your own CSV dataset with the same structure).
4. **Navigate to the project folder**:
   ```bash
   cd path\to\your\project
   ```
5. **Run the Streamlit app**:
   ```bash
   python -m streamlit run heart_disease_app.py
   ```
6. The app will open in your browser at `http://localhost:8501`.

## 📊 Model Performance

- **Training Accuracy:** *[insert value]*
- **Test Accuracy:** *[insert value]*
- **Confusion Matrix:** Displayed in the app for a detailed evaluation.

## 📁 Dataset

The dataset used is the **Heart Disease dataset**, containing health attributes such as:
- `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (maximum heart rate), and others.
- The target column (`target`) indicates whether the person has heart disease (`1`) or not (`0`).

## 📌 Conclusion

This project demonstrates the application of machine learning classifiers to predict heart disease, providing early insights for preventive health measures. The interactive web app allows users to enter their own health data and get real-time predictions, helping individuals assess their risk of heart disease.

---

Let me know if you need anything else!
