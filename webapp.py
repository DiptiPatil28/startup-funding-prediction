
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





st.set_page_config(page_title="Startup Funding Prediction", layout="centered")

st.title("🚀 Startup Funding Prediction Web App")
st.write("Predict whether a startup is likely to receive **High Funding** or **Low Funding**.")





uploaded_file = st.file_uploader("Upload startup funding CSV", type="csv")

if uploaded_file is None:
    st.warning("Please upload the CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)







for col in ['Sr No', 'Remarks']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)


categorical_cols = [
    'Industry Vertical',
    'SubVertical',
    'City  Location',
    'Investors Name'
]



categorical_cols = [c for c in categorical_cols if c in df.columns]
df[categorical_cols] = df[categorical_cols].fillna('Unknown')


df['Amount in USD'] = (
    df['Amount in USD']
    .astype(str)
    .str.replace(',', '', regex=True)
)

df['Amount in USD'] = pd.to_numeric(df['Amount in USD'], errors='coerce')
df = df.dropna(subset=['Amount in USD'])


if 'Date dd/mm/yyyy' in df.columns:
    df['Date dd/mm/yyyy'] = pd.to_datetime(
        df['Date dd/mm/yyyy'], dayfirst=True, errors='coerce'
    )
    df['Year'] = df['Date dd/mm/yyyy'].dt.year
    df['Month'] = df['Date dd/mm/yyyy'].dt.month
    df.drop(columns=['Date dd/mm/yyyy'], inplace=True)




st.subheader("📊 Funding Amount Distribution")

fig1, ax1 = plt.subplots()
sns.histplot(df['Amount in USD'], bins=50, kde=True, ax=ax1)
st.pyplot(fig1)




threshold = df['Amount in USD'].median()
df['High_Funding'] = (df['Amount in USD'] > threshold).astype(int)




X = df.drop(columns=['Amount in USD', 'High_Funding'])
y = df['High_Funding']




X = X.fillna(0)
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)




y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text(classification_report(y_test, y_pred))

fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax2)
st.pyplot(fig2)




st.subheader("🧾 Enter Startup Details")

startup_name = st.text_input("Startup Name", "DemoTech")
industry = st.text_input("Industry Vertical", "FinTech")
subvertical = st.text_input("SubVertical", "Payments")
city = st.text_input("City Location", "Bangalore")
investor = st.text_input("Investor Name", "Sequoia")
year = st.number_input("Year", 2010, 2030, 2024)
month = st.slider("Month", 1, 12, 5)




if st.button("Predict Funding"):

    sample = {
        'Startup Name': startup_name,
        'Industry Vertical': industry,
        'SubVertical': subvertical,
        'City  Location': city,
        'Investors Name': investor,
        'Year': year,
        'Month': month
    }

    sample_df = pd.DataFrame([sample])
    sample_df = pd.get_dummies(sample_df)
    sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

    pred = model.predict(sample_df)[0]
    prob = model.predict_proba(sample_df)[0]

    st.success("High Funding" if pred == 1 else "Low Funding")

    fig3, ax3 = plt.subplots()
    sns.barplot(x=["Low", "High"], y=prob, ax=ax3)
    ax3.set_ylim(0, 1)
    st.pyplot(fig3)
