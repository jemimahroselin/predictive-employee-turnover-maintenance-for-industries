import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(page_title="AI Employee Retention System", layout="wide")

# Background Image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1552664730-d307ca884978");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .title {
        text-align:center;
        font-size:40px;
        color:white;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">AI Employee Retention Intelligence System</p>', unsafe_allow_html=True)

# -------------------------------
# LOAD DATASET
# -------------------------------

df = pd.read_csv("HR-Employee-Attrition.csv")

df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)

# Encode categorical columns
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# Features
X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# EMPLOYEE PREDICTION SECTION
# -------------------------------

st.header("Employee Attrition Analysis")

emp_id = st.number_input("Enter Employee Number", min_value=1)

if st.button("Analyze Employee"):

    emp = X[X['EmployeeNumber'] == emp_id]

    if emp.empty:

        st.error("Employee not found")

    else:

        prediction = model.predict(emp)
        probability = model.predict_proba(emp)

        risk = probability[0][1] * 100

        # Risk Level
        if risk > 70:
            level = "High Risk"
        elif risk > 40:
            level = "Medium Risk"
        else:
            level = "Low Risk"

        col1, col2 = st.columns(2)

        col1.metric("Attrition Risk Score", f"{risk:.2f}%")
        col2.metric("Risk Level", level)

        if prediction[0] == 1:
            st.error("Employee likely to LEAVE the company")
        else:
            st.success("Employee likely to STAY")

        # -------------------------------
        # REASONS
        # -------------------------------

        st.subheader("Possible Reasons for Leaving")

        overtime = df.loc[df['EmployeeNumber']==emp_id,'OverTime'].values[0]
        satisfaction = df.loc[df['EmployeeNumber']==emp_id,'JobSatisfaction'].values[0]
        worklife = df.loc[df['EmployeeNumber']==emp_id,'WorkLifeBalance'].values[0]
        income = df.loc[df['EmployeeNumber']==emp_id,'MonthlyIncome'].values[0]

        reasons = []

        if overtime == 1:
            reasons.append("High overtime workload")

        if satisfaction <= 2:
            reasons.append("Low job satisfaction")

        if worklife <= 2:
            reasons.append("Poor work-life balance")

        if income < 4000:
            reasons.append("Low salary")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No major risk factors detected")

        # -------------------------------
        # HR SUGGESTIONS
        # -------------------------------

        st.subheader("HR Recommendations")

        st.write("• Reduce excessive overtime")
        st.write("• Improve employee recognition programs")
        st.write("• Offer flexible work schedules")
        st.write("• Provide salary increments or incentives")

# -------------------------------
# TOP 5 EMPLOYEES LIKELY TO LEAVE
# -------------------------------

st.header("Top 5 Employees Likely to Leave")

probabilities = model.predict_proba(X)

risk_scores = probabilities[:,1]

df['RiskScore'] = risk_scores

top5 = df.sort_values(by='RiskScore', ascending=False).head(5)

for index, row in top5.iterrows():

    st.subheader(f"Employee ID: {int(row['EmployeeNumber'])}")

    st.write("Attrition Risk Score:", round(row['RiskScore']*100,2), "%")

    st.write("Possible Reasons:")

    if row['OverTime'] == 1:
        st.write("• High overtime workload")

    if row['JobSatisfaction'] <= 2:
        st.write("• Low job satisfaction")

    if row['WorkLifeBalance'] <= 2:
        st.write("• Poor work-life balance")

    if row['MonthlyIncome'] < 4000:
        st.write("• Low salary")

    st.markdown("---")
