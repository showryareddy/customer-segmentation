import streamlit as st
import pickle

kmeans = pickle.load(open("kmeans_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca_model.pkl", "rb"))

st.title("Customer Segmentation App")

income = st.number_input("Income")
spending = st.number_input("Spending Score")

if st.button("Predict Segment"):

    data = [[income, spending]]

    scaled = scaler.transform(data)

    reduced = pca.transform(scaled)

    prediction = kmeans.predict(reduced)

segment_names = {
    0: "Budget Customer",
    1: "Premium Customer",
    2: "Regular Customer"
}


st.success(f"Customer Segment: {segment_names[prediction[0]]}")
