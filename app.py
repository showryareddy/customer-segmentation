import streamlit as st
import pickle

# Load models
kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pcagp_model.pkl", "rb"))

# Title
st.title("Customer Segmentation App")

# Inputs
income = st.number_input("Income")
spending = st.number_input("Total Spending")
children = st.number_input("Children")
tenure = st.number_input("Tenure Days")
webvisits = st.number_input("Web Visits Per Month")
education = st.number_input("Education (Encoded Number)")

# Predict button
if st.button("Predict Segment"):

    # Input data
    data = [[income, spending, children, tenure, webvisits, education]]
    # Scale data
    scaled = scaler.transform(data)

    # PCA transform
    reduced = pca.transform(scaled)

    # Predict cluster
    prediction = kmeans.predict(reduced)

    # Cluster names
    segment_names = {
        0: "Budget Customer",
        1: "Premium Customer",
        2: "Regular Customer",
        3: "VIP Customer"
    }

    # Show result
    st.success(f"Customer Segment: {segment_names[prediction[0]]}")
