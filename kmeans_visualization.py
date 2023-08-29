import pandas as pd
import numpy as np
from ast import literal_eval
from tensorflow import keras
from sklearn.cluster import KMeans
import mlflow.keras
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
 
model = mlflow.keras.load_model("XXX")
model.trainable = False
bottle_neck = model.model_encoder(train_df).numpy()

def kmeans(df_emb, bottle_neck_data,n_clusters):
    train_df = np.stack(df_emb['ada_v2'].values) 
    n_clusters = n_clusters
    estimator = KMeans(n_clusters=n_clusters,random_state=24)  
    estimator.fit(bottle_neck_data)
    return estimator

def get_label(df_emb,kmeans):
    labels = kmeans.labels_
    df_emb['label'] = labels
    df_label = df_emb
    return df_label

def dime_reduct(bottle_neck_data,n_components):
    x_reduced = PCA(n_components=n_components).fit_transform(bottle_neck_data)
    return x_reduced

def plot_kmeans(df_label,x_reduced,palette):
    x = x_reduced[:,0]
    y = x_reduced[:,1]
    plt.figure(figsize=(10,8))
    sns.scatterplot(x,y, hue=df_label['label'], palette=palette)
    plt.show()

def plot_elbow(bottle_neck_data):
    SSE = []  # 存放每次结果的误差平方和
    for k in range(1,9):
        Kmeans = KMeans(n_clusters=k,random_state=24)  # 构造聚类器
        Kmeans.fit(bottle_neck_data)
        SSE.append(Kmeans.inertia_) # estimator.inertia_获取聚类准则的总和
    X = range(1,9)
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(X,SSE,'o-')
    plt.show()

def plot_3d(x_reduced,df_label):
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    fig.add_axes(ax)

    x = x_reduced[::,0]
    y = x_reduced[::,1]
    z = x_reduced[::,2]

    ax.scatter(x, y, z, c=df_label.label, marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

sns.set_style("whitegrid", {'axes.grid' : False})
kmeans = kmeans(df, bottle_neck, 4)
df_label = get_label(df, kmeans)
x_reduced = dime_reduct(bottle_neck,3)
plot_kmeans(df_label,x_reduced,palette=['g','r','c','m'])
plot_elbow(bottle_neck)
plot_3d(x_reduced,df_label)