import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def runModel(dataset: pd.DataFrame) -> tuple:
    """KMeans clustering model

    Args:
        dataset (pd.DataFrame): dataset to run the model on

    Returns:
        tuple: model, x_train, x_test, y_train, y_test
    """
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create KMeans object
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Fit the model
    kmeans.fit(X_train)
    return kmeans, X_train, X_test, y_train, y_test


def plotClusters(kmeans, X_train:np.ndarray, y_train: np.ndarray, xcol: str, ycol: str) -> plt.figure:
    """ plot the clusters

    Args:
        kmeans (kmeans Model): result of the kmeans model
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        xcol (str): column to plot on x-axis
        ycol (str): column to plot on y-axis

    Returns:
        plt.figure: resulting plot
    """
    #convert X_train to a dataframe
    columns = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
    X_train = pd.DataFrame(X_train, columns=columns)
    fig, ax = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    labels = ['Virginica', 'Setosa', 'Versicolor']
    # convert kmeans.labels_ to list of strings
    kmeans.labels_ = [labels[i] for i in kmeans.labels_]
    # put ax as attribute to the scatterplot function
    sns.scatterplot(x=xcol, y=ycol,
                    data=X_train, hue=kmeans.labels_,
                    hue_order=labels,
                    palette='Set2', ax=ax[0],
                    )

    sns.scatterplot(x=kmeans.cluster_centers_[:, columns.index(xcol)],
                    y=kmeans.cluster_centers_[:, columns.index(ycol)],
                    s=100, color='red', label='Centroids', ax=ax[0])
    ax[0].set_title('Training data predictions')
    
    # plot the ground truth clusters
    sns.scatterplot(x=xcol, y=ycol,
                    data=X_train, hue=y_train,
                    hue_order=labels,
                    palette='Set2', ax=ax[1])
    ax[1].set_title('Ground truth clusters')
    return fig
