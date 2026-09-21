# DeliveryETA
Food Delivery Time Prediction
A logistics company needs to tell customers how long a delivery will take. This project builds a regression model that predicts the total delivery time, from order to reception, using courier information, restaurant and customer GPS coordinates, weather, traffic, vehicle type, and order type.

The project covers data cleaning, exploratory analysis, feature engineering (e.g. GPS distance, time-based features), training and comparison of several regression models, hyperparameter tuning, and evaluation with MAE, RMSE, R², and adjusted R². The final model is saved with its preprocessing in a scikit-learn Pipeline and served through a Streamlit app, where a non-technical user enters the delivery details and gets an estimated time in minutes.

Tech stack: Python, pandas, scikit-learn, Streamlit, joblib
