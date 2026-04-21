import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Economic Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_excel("Mad_1.xlsx")
df = pd.read_excel("Mad_2.xlsx")

# CLEANING
df = df.dropna(axis=1, how='all')
df.columns = df.iloc[3]
df = df.iloc[4:]
df = df.rename(columns={df.columns[0]: "Item"})

# Convert numeric
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ---------------- REMOVE UNWANTED ROWS ----------------
# Keep only rows up to "Per Capita PFCE (₹)"
end_index = df[df["Item"].astype(str).str.contains("PFCE", case=False, na=False)].index

if len(end_index) > 0:
    df = df.loc[:end_index[0]]

# Drop empty rows
df = df.dropna(subset=df.columns[1:], how='all')
df = df.reset_index(drop=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Controls")
option = st.sidebar.selectbox(
    "Select Indicator",
    df["Item"].dropna().unique()
)

# ---------------- TITLE ----------------
st.title("📊 Economic Dashboard")
st.markdown("### Indicators Analysis")

# ---------------- DATA ----------------
data = df[df["Item"] == option].iloc[0]
years = df.columns[1:]
values = pd.to_numeric(data[1:], errors='coerce')

# Remove NaN safely
mask = ~np.isnan(values)
values = values[mask]
years = years[mask]

# ---------------- KPI ----------------
if len(values) > 0:
    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Value", round(values.iloc[-1],2))
    col2.metric("Average", round(values.mean(),2))
    col3.metric("Growth", round(values.iloc[-1] - values.iloc[0],2))

# ---------------- GRAPH ----------------
st.subheader("📈 Trend Analysis")

fig, ax = plt.subplots()
ax.plot(years, values, marker='o')
plt.xticks(rotation=45)
plt.grid()

st.pyplot(fig)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Prediction")

X = np.array(range(len(values))).reshape(-1,1)
y = values.values

model = LinearRegression()
model.fit(X, y)

future = model.predict([[len(X)]])

st.success(f"Predicted Next Year Value: {round(float(future),2)}")
