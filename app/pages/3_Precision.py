import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split

from deliveryeta.paths import DELIVERY_DATA_G, MODEL_DIR
from deliveryeta.predict import _load_model

st.set_page_config(page_title="Model Precision", page_icon="🎯")
st.title("Model Precision")

st.caption(
    "Predictions below come from a fresh 80/20 split of the gold dataset (random_state=42), "
    "not the exact split used during training — results will be close to, but not identical "
    "to, the notebook's reported MAE/RMSE/R²."
)


@st.cache_data
def load_split():
    df = pd.read_csv(DELIVERY_DATA_G)
    target_col = "Time_taken(min)"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


X_test, y_test = load_split()
model = _load_model()
y_pred = model.predict(X_test)

results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
results["Residual"] = results["Actual"] - results["Predicted"]

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{results['Residual'].abs().mean():.2f} min")
col2.metric("RMSE", f"{(results['Residual'] ** 2).mean() ** 0.5:.2f} min")
col3.metric("R²", f"{1 - results['Residual'].var() / results['Actual'].var():.4f}")

st.subheader("Predicted vs Actual")
fig1 = px.scatter(results, x="Actual", y="Predicted", opacity=0.4)
fig1.add_shape(
    type="line",
    x0=results["Actual"].min(), y0=results["Actual"].min(),
    x1=results["Actual"].max(), y1=results["Actual"].max(),
    line=dict(color="red", dash="dash"),
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Residuals")
st.plotly_chart(px.histogram(results, x="Residual", nbins=40), use_container_width=True)
st.plotly_chart(px.scatter(results, x="Predicted", y="Residual", opacity=0.4), use_container_width=True)
