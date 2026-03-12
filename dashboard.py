import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Background Image
# ---------------------------

def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
        background-image: url("https://images.unsplash.com/photo-1497366216548-37526070297c");
        background-size: cover;
        background-position: center;
        }

        .block-container{
        background-color: rgba(0,0,0,0.65);
        padding: 2rem;
        border-radius: 10px;
        color:white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

st.title("AI Employee Retention Intelligence System")

st.write("Predict employee turnover risk and help HR take action")

# ---------------------------
# Load Dataset
# ---------------------------

df = pd.read_csv("HR-Employee-Attrition.csv")

df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Features and target
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# ---------------------------
# Employee Prediction
# ---------------------------

st.header("Employee Risk Prediction")

emp_id = st.number_input("Enter Employee Number", min_value=1)

if st.button("Analyze Employee"):

    emp = X[X['EmployeeNumber']==emp_id]

    if emp.empty:

        st.error("Employee not found in dataset")

    else:

        prediction = model.predict(emp)

        probability = model.predict_proba(emp)

        risk = probability[0][1]*100

        st.subheader("Attrition Risk Score")

        st.write(str(round(risk,2)) + " %")

        if prediction[0] == 1:

            st.error("Employee likely to LEAVE the company")

            overtime = df.loc[df['EmployeeNumber']==emp_id,'OverTime'].values[0]
            satisfaction = df.loc[df['EmployeeNumber']==emp_id,'JobSatisfaction'].values[0]
            worklife = df.loc[df['EmployeeNumber']==emp_id,'WorkLifeBalance'].values[0]
            income = df.loc[df['EmployeeNumber']==emp_id,'MonthlyIncome'].values[0]

            st.subheader("Possible Reasons")

            if overtime == 1:
                st.write("• High overtime workload")

            if satisfaction <= 2:
                st.write("• Low job satisfaction")

            if worklife <= 2:
                st.write("• Poor work-life balance")

            if income < 4000:
                st.write("• Low salary")

            st.subheader("HR Recommendations")

            st.write("• Reduce overtime workload")
            st.write("• Improve employee recognition")
            st.write("• Provide flexible working hours")
            st.write("• Consider salary increment")

        else:

            st.success("Employee likely to STAY in the company")

            st.write("Maintain engagement and growth opportunities")

# ---------------------------
# Top Risk Employees
# ---------------------------

st.header("Employees with Highest Attrition Risk")

probabilities = model.predict_proba(X)

risk_scores = probabilities[:,1]

df['RiskScore'] = risk_scores

top5 = df.sort_values(by='RiskScore', ascending=False).head(5)

st.write(top5[['EmployeeNumber','RiskScore']])