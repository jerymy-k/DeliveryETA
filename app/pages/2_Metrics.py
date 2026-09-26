import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Metrics", page_icon="📈")
st.title("Model Metrics")

st.subheader("Model comparison (before tuning)")
comparison = pd.DataFrame(
    {
        "Model": ["RandomForest", "XGBoost", "GradientBoosting", "LinearRegression"],
        "R2": [0.8186, 0.8184, 0.7629, 0.5780],
    }
).sort_values("R2", ascending=False)
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.caption("XGBoost selected for tuning: near-identical R² to RandomForest, ~15-20x faster to train.")

st.subheader("Final tuned model — XGBoost")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", "3.10 min")
col2.metric("RMSE", "3.90 min")
col3.metric("R² (test)", "0.8236")
col4.metric("R² (train)", "0.8688")

st.caption("Train/test gap is mild — not overfit.")

st.subheader("Best hyperparameters")
st.json(
    {
        "learning_rate": 0.05,
        "max_depth": 8,
        "n_estimators": 200,
        "subsample": 1.0,
    }
)
