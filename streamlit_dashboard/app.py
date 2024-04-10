import streamlit as st
from utils import runModel, plotClusters
import pandas as pd

df = pd.read_csv('data/iris.csv')
kmeans, X_train, X_test, y_train, y_test = runModel(df)



def main():
    st.title('Dashboard from Iris dataset.')

    menu = ["Intro analysis", "Model", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Intro analysis":
        st.subheader("Iris dataset")
        # dispalay dataframe
        st.dataframe(df.head())

    elif choice == "Model":
        st.subheader("KMeans Clustering Model")
        # plot the clusters using the plot_clusters function
        fig = plotClusters(kmeans, X_train, y_train)
        st.pyplot(fig)
        
    elif choice == "Prediction":
        st.subheader("Input your measeurements to predict the cluster.")
        # create a form to input the measurements
        sepal_length = st.number_input("Sepal Length")
        sepal_width = st.number_input("Sepal Width")
        petal_length = st.number_input("Petal Length")
        petal_width = st.number_input("Petal Width")

        # create a numpy array from the inputs
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

        # predict the cluster
        prediction = kmeans.predict(input_data)

        # display the cluster in bigger font
        st.subheader(f"The cluster is {prediction[0]}")

if __name__ == "__main__":
    main()