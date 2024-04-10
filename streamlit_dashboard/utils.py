import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def runModel(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create KMeans object
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Fit the model
    kmeans.fit(X_train)
    return kmeans, X_train, X_test, y_train, y_test


def plotClusters(kmeans, X_train:np.ndarray, y_train: np.ndarray):
    #convert X_train to a dataframe
    X_train = pd.DataFrame(X_train, columns=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'])
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)

    # put ax as attribute to the scatterplot function
    sns.scatterplot(x='sepal.length', y='sepal.width',
                    data=X_train, ax=ax[0])
    sns.scatterplot(x='sepal.length', y='sepal.width',
                    data=X_train, hue=kmeans.labels_,
                    palette='viridis', ax=ax[0])
    sns.scatterplot(x=kmeans.cluster_centers_[:, 0],
                    y=kmeans.cluster_centers_[:, 1],
                    s=100, color='red', label='Centroids', ax=ax[0])
    
    # plot the ground truth clusters
    sns.scatterplot(x='sepal.length', y='sepal.width',
                    data=X_train, hue=y_train,
                    palette='viridis', ax=ax[1])
    return fig
