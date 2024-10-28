import numpy as np

# ex1
np.random.seed(0)

def getDataset(meanPoints, meanNoise, stdPoints, stdNoise, N, d):
    params = [2, 1]
    if d == 2:
        params += [3]
    params = np.array(params)
    if d == 1:
        points = np.random.normal(meanPoints, stdPoints, (N, 1))
    else:
        points = np.random.multivariate_normal(meanPoints, stdPoints, N)
    e = np.random.normal(meanNoise, stdNoise, size=N)
    X = np.concatenate((points, np.ones((N, 1))), 1)
    y = X @ params + e 
    
    return points, y, params


def compute_leverage(X):
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    
    H = U @ U.T
    
    leverage = np.diag(H)
    
    return leverage

import matplotlib.pyplot as plt       
from pathlib import Path

file1D = Path("./ex11D.pdf")
file2D = Path("./ex12D.pdf")
if not file1D.is_file() or not file2D.is_file():
    N = 200
    for d in range(1, 3):
        meanPoints = np.zeros(d)
        meanNoise = 0
        fig, axs = plt.subplots(nrows=2, ncols=2, subplot_kw=dict(projection="3d") if d == 2 else None)
        col, row = 0, 0
        for (sigmaPoints, sigmaNoise) in zip([1, 5, 1, 5], [1, 1, 5, 5]):
            stdPoints = np.array(sigmaPoints) 
            stdNoise = np.array(sigmaNoise)
            if d == 2:
                stdPoints = sigmaPoints * np.eye(2)
            data, y, params = getDataset(meanPoints, meanNoise, stdPoints, stdNoise, N, d)
            leverage_scores = compute_leverage(data)
            
            outlier_idx = np.abs(leverage_scores - leverage_scores.mean()) > 3 * leverage_scores.std()
            
            if d == 1:
                axs[row, col].scatter(data[outlier_idx == False], y[outlier_idx == False], color='blue')
                axs[row, col].scatter(data[outlier_idx], y[outlier_idx], color='red')   
            else:
                axs[row, col].scatter(data[outlier_idx == False][:, 0], data[outlier_idx == False][:, 1], y[outlier_idx == False], color='blue')
                axs[row, col].scatter(data[outlier_idx][:, 0], data[outlier_idx][:, 1], y[outlier_idx], color='red')
    
            xs = np.expand_dims(np.array(axs[row, col].get_xlim()), axis=1)
            ys = np.expand_dims(np.array(axs[row, col].get_xlim()), axis=1)
            coords = [xs] if d == 1 else [xs, ys]
            zs = np.hstack([*coords, np.ones(shape=xs.shape)]) @ params
            coords = [xs.squeeze(), zs] if d == 1 else [xs.squeeze(), ys.squeeze(), zs]
            axs[row, col].plot(*coords, color="black")
            
            col += 1
            if col >= 2:
                row += 1
                col = 0
            fig.savefig("ex1" + str(d) + "D" + ".pdf")
                     
# ex2
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

file_ex2 = Path("./ex2_1.pdf")
if not file_ex2.is_file():
    X_train, X_test, y_train, y_test = generate_data_clusters(400, 200, 2, 2, 0.1)
    for n in [1, 3, 5, 7, 11]:
        knn = KNN(contamination=0.1, n_neighbors=n)
        knn.fit(X_train)
        train_pred = knn.predict(X_train)
        test_pred = knn.predict(X_test)
        
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue')
        axs[0, 0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red')
        axs[0, 0].set_title("Ground Truth Train")
        
        axs[0, 1].scatter(X_train[train_pred == 0][:, 0], X_train[train_pred == 0][:, 1], color='blue')
        axs[0, 1].scatter(X_train[train_pred == 1][:, 0], X_train[train_pred == 1][:, 1], color='red')
        axs[0, 1].set_title("Predicted Train")
        
        axs[1, 0].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], color='blue')
        axs[1, 0].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red')
        axs[1, 0].set_title("Ground Truth Test")
        
        axs[1, 1].scatter(X_test[test_pred == 0][:, 0], X_test[test_pred == 0][:, 1], color='blue')
        axs[1, 1].scatter(X_test[test_pred == 1][:, 0], X_test[test_pred == 1][:, 1], color='red')
        axs[1, 1].set_title("Predicted Test")
        
        fig.suptitle("KNN for " + str(n) + " neighbors BA Train: " + str(balanced_accuracy_score(y_train, train_pred)) + " BA Test: " + str(balanced_accuracy_score(y_test, test_pred)))
        plt.savefig("ex2_" + str(n) + ".pdf")
    
# ex3
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs

file_ex3 = Path("./ex3.pdf")
if not file_ex3.is_file():
    data, _ = make_blobs(n_samples=[200, 100], n_features=2, centers=[[-10, -10], [10, 10]], cluster_std=[2, 6])
    knn = KNN(contamination=0.07)
    lof = LOF(contamination=0.07)

    knn.fit(data)
    lof.fit(data)

    knn_preds = knn.predict(data)
    lof_preds = lof.predict(data)
    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].scatter(data[knn_preds == 0][:, 0], data[knn_preds == 0][:, 1], color='blue')
    axs[0].scatter(data[knn_preds == 1][:, 0], data[knn_preds == 1][:, 1], color='red')
    axs[1].scatter(data[lof_preds == 0][:, 0], data[lof_preds == 0][:, 1], color='blue')
    axs[1].scatter(data[lof_preds == 1][:, 0], data[lof_preds == 1][:, 1], color='red')
    axs[0].set_title("KNN")
    axs[1].set_title("LOF")
    fig.savefig("ex3.pdf")

# ex4
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer

d = loadmat("./cardio.mat")
data, y = d['X'], d['y']
X_train, X_test, y_train, y_test = train_test_split(data, y)
X_train, X_test = standardizer(X_train, X_test)

c = (sum(y_train == 1) / y_train.size)[0]
knn_scores = []
lof_scores = []
for n in range(30, 121, 10):
    knn = KNN(n_neighbors=n, contamination=c)
    lof = LOF(n_neighbors=n, contamination=c)
    knn.fit(X_train)
    lof.fit(X_train)
    knn_preds = knn.predict(X_test)
    lof_preds = lof.predict(X_test)
    
    ba_knn = balanced_accuracy_score(y_test, knn_preds)
    ba_lof = balanced_accuracy_score(y_test, lof_preds)
    
    knn_score = knn.decision_function(X_test)
    lof_score = lof.decision_function(X_test)
 
    knn_scores.append(knn_score)
    lof_scores.append(lof_score)
        
    print("BA KNN " + str(n) + " : " + str(ba_knn))
    print("BA LOF " + str(n) + " : " + str(ba_lof))

knn_scores = np.array(knn_scores).T
lof_scores = np.array(lof_scores).T

knn_scores = standardizer(knn_scores)
lof_scores = standardizer(lof_scores)

from pyod.models.combination import average, maximization

knn_averaged, knn_maximized = average(knn_scores), maximization(knn_scores)
lof_averaged, lof_maximized = average(lof_scores), maximization(lof_scores)

def getPredsFromScore(scores):
    threshold = np.quantile(scores, 1 - c)
    preds = np.zeros(scores.shape[0])
    preds[scores <= threshold] = 0
    preds[scores > threshold] = 1
    
    return preds

knn_averaged_preds = getPredsFromScore(knn_averaged)
knn_maximized_preds = getPredsFromScore(knn_maximized)
lof_averaged_preds = getPredsFromScore(lof_averaged)
lof_maximized_preds = getPredsFromScore(lof_maximized)
ba_knn_averaged = balanced_accuracy_score(y_test, knn_averaged_preds)
ba_knn_maximized = balanced_accuracy_score(y_test, knn_maximized_preds)
ba_lof_averaged = balanced_accuracy_score(y_test, lof_averaged_preds)
ba_lof_maximized= balanced_accuracy_score(y_test, lof_maximized_preds)

print("BA KNN averaged= ", ba_knn_averaged)
print("BA KNN maximized= ", ba_knn_maximized)
print("BA LOF averaged= ", ba_lof_averaged)
print("BA LOF maximized= ", ba_lof_maximized)

    