
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Sales Prediction Studio",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/best_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

st.title("📈 Sales Prediction Studio")
st.write("Predict sales from advertising expenditure.")

try:
    model, features = load_model()
    data = pd.read_csv("Advertising.csv")
except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()

st.sidebar.header("Advertising Budget")

tv = st.sidebar.slider("TV advertising", 0.0, 300.0, 150.0)
radio = st.sidebar.slider("Radio advertising", 0.0, 50.0, 25.0)
newspaper = st.sidebar.slider(
    "Newspaper advertising", 0.0, 120.0, 30.0
)

if st.sidebar.button("Predict Sales", type="primary"):
    inputs = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=features
    )
    prediction = float(model.predict(inputs)[0])
    st.session_state["prediction"] = prediction

st.subheader("Sales Prediction")

if "prediction" in st.session_state:
    st.metric(
        "Predicted Sales",
        f'{st.session_state["prediction"]:.2f}'
    )
else:
    st.info("Adjust the budgets and click Predict Sales.")

st.divider()

st.subheader("Business Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", f"{len(data):,}")
c2.metric("Average Sales", f'{data["Sales"].mean():.2f}')
c3.metric("Total Advertising Spend", 
          f'{data[features].sum().sum():,.2f}')

st.subheader("Advertising Impact")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, feature in zip(axes, features):
    ax.scatter(data[feature], data["Sales"], alpha=0.65)
    ax.set_xlabel(feature)
    ax.set_ylabel("Sales")
    ax.set_title(f"{feature} vs Sales")
    ax.grid(alpha=0.2)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Model Performance")

try:
    results = pd.read_csv("model_results.csv")
    st.dataframe(results, use_container_width=True, hide_index=True)
except FileNotFoundError:
    st.info("Run train_model.py to generate model metrics.")

st.subheader("Dataset Preview")
st.dataframe(data.head(20), use_container_width=True)
