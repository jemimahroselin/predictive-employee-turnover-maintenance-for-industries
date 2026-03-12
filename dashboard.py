import streamlit as st
import pandas as pd
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Employee Retention System",
    page_icon="💼",
    layout="wide"
)

# --------------------------------------------------
# DARK BACKGROUND IMAGE WITH GRADIENT
# --------------------------------------------------

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.85)),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        h1, h2, h3, h4, h5, p, label, div {{
        color: white !important;
        }}

        .stNumberInput input {{
        background-color: rgba(0,0,0,0.6);
        color: white;
        border-radius: 8px;
        }}

        .stButton>button {{
        background-color:#ff4b4b;
        color:white;
        border-radius:10px;
        height:45px;
        width:220px;
        font-size:18px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Using your background image
add_bg_from_local("company_bg.jpg")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("HR-Employee-Attrition.csv")

df['Attrition'] = df['Attrition'].map({'Yes':1,'No':0})

df = df.drop(['EmployeeCount','Over18','StandardHours'], axis=1)

# --------------------------------------------------
# ENCODE DATA
# --------------------------------------------------

label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = label_encoder.fit_transform(df[col])

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

X = df.drop(['Attrition'], axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("AI Employee Retention System")

st.write(
"Predict whether an employee will stay or leave the company and provide HR suggestions."
)

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

emp_id = st.number_input("Enter Employee Number", step=1)

if st.button("Predict Employee Status"):

    emp = X[X['EmployeeNumber'] == emp_id]

    if emp.empty:

        st.error("Employee not found in dataset")

    else:

        prediction = model.predict(emp)
        probability = model.predict_proba(emp)

        risk = probability[0][1] * 100

        st.subheader("Attrition Risk Score")

        st.write(f"{round(risk,2)} %")

        if prediction[0] == 1:

            st.error("Prediction: Employee likely to LEAVE")

            st.subheader("Reasons & HR Suggestions")

            overtime = df.loc[df['EmployeeNumber']==emp_id,'OverTime'].values[0]
            satisfaction = df.loc[df['EmployeeNumber']==emp_id,'JobSatisfaction'].values[0]
            worklife = df.loc[df['EmployeeNumber']==emp_id,'WorkLifeBalance'].values[0]
            income = df.loc[df['EmployeeNumber']==emp_id,'MonthlyIncome'].values[0]

            if overtime == 1:
                st.write("- High overtime → Reduce workload")

            if satisfaction <= 2:
                st.write("- Low job satisfaction → Improve recognition")

            if worklife <= 2:
                st.write("- Poor work life balance → Flexible schedule")

            if income < 4000:
                st.write("- Low salary → Consider salary increment")

        else:

            st.success("Prediction: Employee likely to STAY")

            st.write("HR Suggestion: Maintain engagement and growth opportunities")

# --------------------------------------------------
# TOP 5 EMPLOYEES LIKELY TO LEAVE
# --------------------------------------------------

st.subheader("Top 5 Employees Likely to Leave")

probabilities = model.predict_proba(X)

risk_scores = probabilities[:,1]

df['RiskScore'] = risk_scores

top5 = df.sort_values(by='RiskScore', ascending=False).head(5)

for index, row in top5.iterrows():

    st.write("Employee ID:", row['EmployeeNumber'])

    st.write("Risk Score:", round(row['RiskScore']*100,2), "%")

    reasons = []

    if row['OverTime'] == 1:
        reasons.append("High overtime")

    if row['MonthlyIncome'] < 4000:
        reasons.append("Low salary")

    if row['JobSatisfaction'] <= 2:
        reasons.append("Low job satisfaction")

    if row['WorkLifeBalance'] <= 2:
        reasons.append("Poor work life balance")

    if len(reasons) == 0:
        reasons.append("General attrition risk")

    st.write("Possible Reasons:", ", ".join(reasons))

    st.write("-----------------------------")
