import streamlit as st

from deliveryeta.predict import predict, PredictionInputError

st.set_page_config(page_title="DeliveryETA", page_icon="🛵")
st.title("Delivery Time Predictor")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
        weather = st.selectbox("Weather conditions", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
        traffic = st.selectbox("Road traffic density", ["Low", "Medium", "High", "Jam"])
        order_type = st.selectbox("Order type", ["Snack", "Meal", "Drinks", "Buffet"])
        vehicle_type = st.selectbox("Vehicle type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
        festival = st.selectbox("Festival day?", ["No", "Yes"])

    with col2:
        city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"])
        courier_age = st.number_input("Courier age", min_value=15, max_value=65, value=30)
        courier_rating = st.slider("Courier rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        vehicle_condition = st.selectbox("Vehicle condition", [3, 2, 1, 0])
        multiple_deliveries = st.selectbox("Multiple deliveries", [0, 1, 2, 3])
        order_hour = st.slider("Order hour", min_value=0, max_value=23, value=12)

    submitted = st.form_submit_button("Predict delivery time")

if submitted:
    payload = {
        "Distance_km": distance_km,
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "Festival": festival,
        "City": city,
        "Delivery_person_Age": courier_age,
        "Delivery_person_Ratings": courier_rating,
        "Vehicle_condition": vehicle_condition,
        "multiple_deliveries": multiple_deliveries,
        "Order_Hour": order_hour,
    }
    try:
        eta = predict(payload)
        st.success(f"Estimated delivery time: **{eta} minutes**")
    except PredictionInputError as e:
        st.error(str(e))