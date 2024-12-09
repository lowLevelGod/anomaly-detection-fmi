import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, InputLayer

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
    plt.bar(range(1, len(eigvals) + 1), sorted_eigenvalues / np.sum(sorted_eigenvalues), alpha=0.7, label="Individual Variances")
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

    def plot_pca_anomalies(anomalies, title):
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw=dict(projection="3d"))
        ax.scatter(*data.T, label="Normal Points", alpha=0.7)
        ax.scatter(*data[anomalies].T, color='red', label="Anomalies")
        fig.suptitle(title)
        ax.legend()
        fig.savefig("ex1.3" + title + ".pdf")
        plt.clf()

    plot_pca_anomalies(anomalies_pc3, title="Anomalies based on PC3")
    plot_pca_anomalies(anomalies_pc2, title="Anomalies based on PC2")
    
    # 4
    normalization_factors = np.sqrt(eigvals)
    normalized_data = pca_data / normalization_factors

    anomaly_scores = np.sum(normalized_data**2, axis=1)

    contamination_rate = 0.1
    threshold = np.quantile(anomaly_scores, 1 - contamination_rate)
    anomalies = anomaly_scores > threshold
    
    def plot_anomalies(data, anomalies, title):
        fig, ax = plt.subplots(1, 1, figsize=(20, 8), subplot_kw=dict(projection="3d"))
        ax.scatter(*data.T, label="Normal Points", alpha=0.7)
        ax.scatter(*data[anomalies].T, color='red', label="Anomalies", alpha=0.7)
        fig.suptitle(title)
        ax.legend()
        fig.savefig("ex1.4.pdf")
        plt.clf()

    plot_anomalies(normalized_data, anomalies, title="Anomalies based on Normalized Distance")
    
def ex2():
    # 1
    
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
    
    # 2
    
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

def ex3():
    # 1
    
    data = loadmat("./shuttle.mat")
    X = data["X"]
    y = data["y"].ravel()  

    train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.5)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    # 2
    
    class Autoencoder(Model):
        def __init__(self):
            super(Autoencoder, self).__init__()
            
            self.encoder = Sequential([
                Dense(8, activation="relu"),
                Dense(5, activation="relu"),
                Dense(3, activation="relu"),
            ])
            
            self.decoder = Sequential([
                Dense(5, activation="relu"),
                Dense(8, activation="relu"),
                Dense(9, activation="sigmoid"),
            ])
        
        def call(self, inputs):
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded

    # 3
    
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer="adam", loss="mse")

    history = autoencoder.fit(
        train_data,
        train_data, 
        epochs=100,
        batch_size=1024,
        validation_data=(test_data, test_data),
        verbose=1
    )

    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("ex3.3.pdf")
    plt.clf()
    
    # 4
    
    train_reconstructions = autoencoder.predict(train_data)
    test_reconstructions = autoencoder.predict(test_data)

    train_errors = np.mean(np.square(train_data - train_reconstructions), axis=1)
    test_errors = np.mean(np.square(test_data - test_reconstructions), axis=1)

    contamination_rate = np.mean(train_labels)
    threshold = np.quantile(train_errors, 1 - contamination_rate)

    train_predictions = train_errors > threshold
    test_predictions = test_errors > threshold
    
    train_ba = balanced_accuracy_score(train_labels, train_predictions)
    test_ba = balanced_accuracy_score(test_labels, test_predictions)

    print(f"Balanced Accuracy - Train: {train_ba:.3f}, Test: {test_ba:.3f}")


def ex4():
    # 1
    
    (train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()

    train_data = train_data.astype("float32") / 255.0
    test_data = test_data.astype("float32") / 255.0

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    noise_factor = 0.35
    train_data_noisy = train_data + noise_factor * tf.random.normal(shape=train_data.shape)
    test_data_noisy = test_data + noise_factor * tf.random.normal(shape=test_data.shape)

    train_data_noisy = tf.clip_by_value(train_data_noisy, 0.0, 1.0)
    test_data_noisy = tf.clip_by_value(test_data_noisy, 0.0, 1.0)
    
    # 2
    
    class ConvAutoencoder(Model):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()
            
            self.encoder = Sequential([
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(8, (3, 3), activation="relu", strides=2, padding="same"),
                Conv2D(4, (3, 3), activation="relu", strides=2, padding="same")
            ])
            
            self.decoder = Sequential([
                Conv2DTranspose(4, (3, 3), activation="relu", strides=2, padding="same"),
                Conv2DTranspose(8, (3, 3), activation="relu", strides=2, padding="same"),
                Conv2D(1, (3, 3), activation="sigmoid", padding="same")
            ])
        
        def call(self, inputs):
            encoded = self.encoder(inputs)
            decoded = self.decoder(encoded)
            return decoded
   
    # 3
    
    autoencoder = ConvAutoencoder()
    autoencoder.compile(optimizer="adam", loss="mse")

    history = autoencoder.fit(
        train_data, train_data, 
        epochs=10,
        batch_size=64,
        validation_data=(test_data, test_data),
        verbose=1
    )
    
    train_data_reconstructed = autoencoder.predict(train_data)
    test_data_reconstructed = autoencoder.predict(test_data)
    test_data_noisy_reconstructed = autoencoder.predict(test_data_noisy)

    train_errors = np.mean(np.square(train_data - train_data_reconstructed), axis=(1, 2, 3))
    test_errors = np.mean(np.square(test_data - test_data_reconstructed), axis=(1, 2, 3))
    test_noisy_errors = np.mean(np.square(test_data_noisy - test_data_noisy_reconstructed), axis=(1, 2, 3))

    threshold = np.mean(train_errors) + np.std(train_errors)

    test_predictions = test_errors > threshold
    test_noisy_predictions = test_noisy_errors > threshold

    test_accuracy = np.mean(test_predictions)  
    test_noisy_accuracy = np.mean(test_noisy_predictions)  
    print(f"Accuracy on original test images: {test_accuracy:.3f}")
    print(f"Accuracy on noisy test images: {test_noisy_accuracy:.3f}")
    
    # 4
    
    def plot_images(original, noisy, reconstructed_original, reconstructed_noisy, title, num_images=5):
        plt.figure(figsize=(15, 10))

        for i in range(num_images):

            plt.subplot(4, num_images, i + 1)
            plt.imshow(tf.squeeze(original[i]), cmap="gray")
            plt.axis("off")

            plt.subplot(4, num_images, i + 1 + num_images)
            plt.imshow(tf.squeeze(noisy[i]), cmap="gray")
            plt.axis("off")

            plt.subplot(4, num_images, i + 1 + 2 * num_images)
            plt.imshow(tf.squeeze(reconstructed_original[i]), cmap="gray")
            plt.axis("off")

            plt.subplot(4, num_images, i + 1 + 3 * num_images)
            plt.imshow(tf.squeeze(reconstructed_noisy[i]), cmap="gray")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(title)
        plt.clf()
        
    plot_images(test_data, test_data_noisy, test_data_reconstructed, test_data_noisy_reconstructed, "ex4.4.pdf")
    
    # 5
    
    autoencoder.compile(optimizer="adam", loss="mse")

    history_denoising = autoencoder.fit(
        train_data_noisy, train_data, 
        epochs=10,
        batch_size=64,
        validation_data=(test_data_noisy, test_data),
        verbose=1
    )

    test_data_noisy_reconstructed_denoising = autoencoder.predict(test_data_noisy)
    plot_images(test_data, test_data_noisy, test_data_reconstructed, test_data_noisy_reconstructed_denoising, "ex4.5.pdf")


  
ex1()
ex2()
ex3()
ex4()