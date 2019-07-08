import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

classes = ['bacon', 'bean', 'beef', 'blender', 'bowl', 'bread', 'butter', 'cabbage', 'carrot', 'celery', 'cheese', 'chicken', 'chickpea', 'corn', 'cream', 'cucumber', 'cup', 'dough', 'egg', 'flour', 'garlic', 'ginger', 'it', 'leaf', 'lemon', 'lettuce', 'lid', 'meat', 'milk', 'mixture', 'mushroom', 'mussel', 'mustard', 'noodle', 'oil', 'onion', 'oven', 'pan', 'paper', 'pasta', 'pepper', 'plate', 'pork', 'pot', 'potato', 'powder', 'processor', 'rice', 'salad', 'salt', 'sauce', 'seaweed', 'sesame', 'shrimp', 'soup', 'squid', 'sugar', 'that', 'them', 'they', 'tofu', 'tomato', 'vinegar', 'water', 'whisk', 'wine', 'wok']
def visfeat(data, label, savepath):
    np.random.seed(1)
    colors = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0], [0.0, 0.9, 0.5], [0.9, 0.5, 0.0], [0.8, 0.2, 0.2]])
    # only look at n classes
    n = 10
    X = np.array(data)
    Y = np.array(label)
    ind = np.where(Y < n)[0]
    Y_spl = Y[ind]
    X_spl = X[ind]
    Y = Y_spl[:]
    X = X_spl[:]
    # X = X/np.linalg.norm(X, 2, 1, True)
    area = np.ones(len(Y))
    area[:n] = 30
    area *=3
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(X)

    ckpt_dir="images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    for i in range(n):
        inds_i = np.where(Y == i)[0]
        X_sub = X_tsne[inds_i]
        # X_sub = X_pca[inds_i]
        plt.scatter(X_sub[:, 0], X_sub[:, 1], s=area[inds_i], c=np.array([colors[i]]), label=classes[i], alpha=0.5)
    plt.legend()
    plt.title('t-SNE')
    plt.subplot(122)
    for i in range(n):
        inds_i = np.where(Y == i)[0]
        X_sub = X_pca[inds_i]
        plt.scatter(X_sub[:, 0], X_sub[:, 1], s=area[inds_i], c=np.array([colors[i]]), label=classes[i], alpha=0.5)
    plt.legend()
    plt.title('PCA')
    plt.savefig(savepath, dpi=120)
