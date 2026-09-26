import pandas as pd
import plotly.express as px
import streamlit as st

from deliveryeta.paths import DELIVERY_DATA_G

st.set_page_config(page_title="Visualisations", page_icon="📊")
st.title("Data Visualisations")


@st.cache_data
def load_data():
    return pd.read_csv(DELIVERY_DATA_G)


df = load_data()
target_col = "Time_taken(min)"

st.subheader("Distribution of delivery time")
st.plotly_chart(px.histogram(df, x=target_col, nbins=40), use_container_width=True)

st.subheader("Delivery time vs distance")
st.plotly_chart(
    px.scatter(df, x="Distance_km", y=target_col, color="Road_traffic_density", opacity=0.4),
    use_container_width=True,
)

st.subheader("Delivery time by traffic density")
st.plotly_chart(
    px.box(df, x="Road_traffic_density", y=target_col, category_orders={"Road_traffic_density": ["Low", "Medium", "High", "Jam"]}),
    use_container_width=True,
)

st.subheader("Correlation with delivery time")
numeric_df = df.select_dtypes(include="number")
corr = numeric_df.corr()[target_col].drop(target_col).sort_values()
st.plotly_chart(px.bar(corr, orientation="h", labels={"value": "Correlation", "index": ""}), use_container_width=True)
