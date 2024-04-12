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
        # write intro text to iris dataset
        st.write("""
        The Iris dataset is a classic dataset in the field of machine learning. It is included in the sklearn library.
        The dataset contains 150 samples of iris flowers. There are three species of iris flowers in the dataset: setosa, versicolor, and virginica.
        Each sample has four features: sepal length, sepal width, petal length, and petal width.
        The goal is to classify the species of iris flowers based on the four features.
        """)
        # dispalay dataframe
        st.dataframe(df.head())

    elif choice == "Model":
        st.subheader("KMeans Clustering Model")
        # plot the clusters using the plot_clusters function
        # select of columns to plot using streamlit
        xcol = st.selectbox("Select the columns to plot on X", df.columns)
        ycol = st.selectbox("Select the columns to plot on Y", df.columns)
        

        fig = plotClusters(kmeans, X_train, y_train, xcol, ycol)
        st.pyplot(fig)
        
    elif choice == "Prediction":
        st.subheader("Input your measeurements to predict the cluster.")
        # create a form to input the measurements
        sepal_length = st.number_input("Sepal Length",
                                       min_value=df['sepal.length'].min()-0.5,
                                       max_value=df['sepal.length'].max()+0.5,
                                       value=df['sepal.length'].mean(),
                                       step=0.1)
        sepal_width = st.number_input("Sepal Width",
                                      min_value=df['sepal.width'].min()-0.5,
                                      max_value=df['sepal.width'].max()+0.5,
                                      value=df['sepal.width'].mean(),
                                      step=0.1)
        petal_length = st.number_input("Petal Length",
                                       min_value=df['petal.length'].min()-0.5,
                                       max_value=df['petal.length'].max()+0.5,
                                       value=df['petal.length'].mean(),
                                       step=0.1)
        petal_width = st.number_input("Petal Width",
                                      min_value=df['petal.width'].min()-0.5,
                                      max_value=df['petal.width'].max()+0.5,
                                      value=df['petal.width'].mean(),
                                      step=0.1)

        # create a numpy array from the inputs
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

        # predict the cluster
        prediction = kmeans.predict(input_data)

        # display the cluster in bigger font
        st.subheader(f"The cluster is {prediction[0]}")

if __name__ == "__main__":
    main()