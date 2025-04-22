import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
st.set_page_config(page_title='Customer Segmentation', page_icon=':star:')
df = pd.read_csv("Mall_Customers.csv")

st.write('# Customer Segmentation Deployment')
st.write('---')
st.subheader('Enter your details to predict your Segment')

age = st.slider('Age:', 0, 100)
income = st.slider('Annual Income (k$):', 0, 100)
spending_score = st.slider('Spending Score (1-100):', 0, 100)

columns_to_scale = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
df_scaled_training = df[columns_to_scale]
kmeans = KMeans(n_clusters=5)  # Set the desired number of clusters
kmeans.fit(df_scaled_training)
# Train the clustering algorithm
kmeans = KMeans(n_clusters=5)  # Set the desired number of clusters
kmeans.fit(df_scaled_training)

def predict_cluster(new_data):
   new_data_array = np.array(new_data).reshape(1, -1)  # Reshape to match the model input
   cluster_label = kmeans.predict(new_data_array)
   return cluster_label[0]

new_data = [ age, income, spending_score]
predicted_cluster = predict_cluster(new_data)  # Use your predict_cluster function here
st.write("Predicted Cluster:", predicted_cluster)
