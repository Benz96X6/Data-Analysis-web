import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Sustainable Waste Management Dashboard")

# โหลดไฟล์ CSV
df = pd.read_csv("oo.csv")

st.write("ตัวอย่างข้อมูล")
st.dataframe(df.head())

# เลือก feature
selected_features = [
    'population',
    'recyclable_kg',
    'organic_kg',
    'collection_capacity_kg',
    'is_weekend',
    'is_holiday',
    'recycling_campaign',
    'temp_c',
    'rain_mm'
]

X = df[selected_features]
y = df['waste_kg']

# clean missing values
df_combined = pd.concat([X, y], axis=1).dropna()
X = df_combined[selected_features]
y = df_combined['waste_kg']

st.subheader("Linear Regression Model")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

st.write("📉 Mean Squared Error (MSE):", mse)
st.write("📈 R-squared (R²):", r2)

# ===== Plot =====
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)

min_val = Y_test.min()
max_val = Y_test.max()
ax.plot([min_val, max_val], [min_val, max_val], '--', color='red', lw=2)

ax.set_xlabel('Actual Waste (kg)')
ax.set_ylabel('Predicted Waste (kg)')
ax.set_title('Predicted vs Actual Waste')
ax.grid(True)

st.pyplot(fig)
