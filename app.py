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

    st.success(f"Customer Segment: {prediction[0]}")
