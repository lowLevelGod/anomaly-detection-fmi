import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from scipy.io import loadmat

def ex1():
    # 1
    
    data = np.random.multivariate_normal(mean=[5, 10, 2], cov=[[3, 2, 2], [2, 10, 1], [2, 1, 2]], size=500)
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw=dict(projection="3d"))
    ax.scatter(*data.T,)
    fig.savefig("ex1.1.pdf")
    plt.clf()
    
    def pca(X):
        X_centered = X - np.mean(X, axis=0)
        covMatrix = X_centered.T @ X_centered
        covMatrix /= X.shape[0]
        
        eigvals, eigvectors = np.linalg.eigh(covMatrix)
        
        return eigvals, eigvectors, X_centered
    # 2
    eigvals, eigvectors, X_centered = pca(data)
    sorted_indices = np.argsort(eigvals)[::-1]
    sorted_eigenvalues = eigvals[sorted_indices]
    sorted_eigenvectors = eigvectors[sorted_indices]
    cumulative_explained_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(eigvals) + 1), eigvals, alpha=0.7, label="Individual Variances")
    plt.step(range(1, len(sorted_eigenvalues) + 1), cumulative_explained_variance, where='mid',
            label="Cumulative Explained Variance", color='orange', linewidth=2)

    plt.title("Explained Variance Plot")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Variance Explained")
    plt.legend()
    plt.grid(True)
    plt.savefig("ex1.2.pdf")
    plt.clf()
    
    # 3
    pca_data = X_centered @ sorted_eigenvectors
    
    def identify_anomalies(pca_data, component_index, contamination_rate=0.1):
        component_values = pca_data[:, component_index]
        mean_value = np.mean(component_values)
        deviations = np.abs(component_values - mean_value)
        
        threshold = np.quantile(deviations, 1 - contamination_rate)
        labels = deviations > threshold 
        
        return labels, threshold

    anomalies_pc3, threshold_pc3 = identify_anomalies(pca_data, component_index=2, contamination_rate=0.1)
    anomalies_pc2, threshold_pc2 = identify_anomalies(pca_data, component_index=1, contamination_rate=0.1)

    def plot_pca_anomalies(pca_data, anomalies, pc_x, pc_y, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, pc_x], pca_data[:, pc_y], label="Normal Points", alpha=0.7)
        plt.scatter(pca_data[anomalies, pc_x], pca_data[anomalies, pc_y], color='red', label="Anomalies", alpha=0.7)
        plt.title(title)
        plt.xlabel(f"Principal Component {pc_x + 1}")
        plt.ylabel(f"Principal Component {pc_y + 1}")
        plt.legend()
        plt.grid(True)
        plt.savefig("ex1.3" + title + ".pdf")
        plt.clf()

    plot_pca_anomalies(pca_data, anomalies_pc3, pc_x=0, pc_y=1, title="Anomalies based on PC3")
    plot_pca_anomalies(pca_data, anomalies_pc2, pc_x=0, pc_y=1, title="Anomalies based on PC2")
    
    # 4
    normalization_factors = np.sqrt(eigvals)
    normalized_data = pca_data / normalization_factors

    anomaly_scores = np.sum(normalized_data**2, axis=1)

    contamination_rate = 0.1
    threshold = np.quantile(anomaly_scores, 1 - contamination_rate)
    anomalies = anomaly_scores > threshold
    
    def plot_anomalies(data, anomalies, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], label="Normal Points", alpha=0.7)
        plt.scatter(data[anomalies, 0], data[anomalies, 1], color='red', label="Anomalies", alpha=0.7)
        plt.title(title)
        plt.xlabel("Transformed Component 1")
        plt.ylabel("Transformed Component 2")
        plt.legend()
        plt.grid(True)
        plt.savefig("ex1.4.pdf")
        plt.clf()

    plot_anomalies(normalized_data, anomalies, title="Anomalies based on Normalized Distance")
    
def ex2():
    mat = loadmat("./shuttle.mat")
    data, labels = mat['X'], mat['y']
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.6, random_state=42)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    contamination_rate = np.sum(train_labels) / len(train_labels)

    pca_model = PCA(contamination=contamination_rate, random_state=42)
    pca_model.fit(train_data)

    explained_variance = pca_model.explained_variance_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label="Individual Variances")
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
            label="Cumulative Explained Variance", color='orange', linewidth=2)
    plt.title("Explained Variance by PCA Components")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Variance Explained")
    plt.legend()
    plt.grid(True)
    plt.savefig("ex2.1.pdf")
    plt.clf()

    y_train_pred = pca_model.predict(train_data) 
    y_test_pred = pca_model.predict(test_data)

    train_ba = balanced_accuracy_score(train_labels, y_train_pred)
    test_ba = balanced_accuracy_score(test_labels, y_test_pred)

    print(f"PCA Balanced Accuracy - Train: {train_ba:.3f}, Test: {test_ba:.3f}")

    kpca_model = KPCA(contamination=contamination_rate, kernel="rbf", random_state=42)
    kpca_model.fit(train_data)

    y_train_pred_kpca = kpca_model.predict(train_data)
    y_test_pred_kpca = kpca_model.predict(test_data)

    train_ba_kpca = balanced_accuracy_score(train_labels, y_train_pred_kpca)
    test_ba_kpca = balanced_accuracy_score(test_labels, y_test_pred_kpca)

    print(f"KPCA Balanced Accuracy - Train: {train_ba_kpca:.3f}, Test: {test_ba_kpca:.3f}")

# ex1()
ex2()