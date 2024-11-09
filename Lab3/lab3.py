from sklearn.datasets import make_blobs

# ex1
# 1.1
def ex1():
    data, _ = make_blobs(n_samples=500, n_features=2, random_state=42, centers=1)
    # 1.2
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    np.random.seed(42)
    proj_vectors = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=5)
    proj_vectors = normalize(proj_vectors)
    proj_data = data @ proj_vectors.T

    left = proj_data.min(axis=0)
    right = proj_data.max(axis=0)

    left -= 0.8 * (right - left)
    right += 0.8 * (right - left)

    def train(n_bins):
        bin_edges = []
        scores = []
        probabilities = []
        histograms = []

        for feature_idx in range(proj_data.shape[-1]):
            feature_data = proj_data[:, feature_idx]
            hist, bins = np.histogram(feature_data, bins=n_bins, range=(left[feature_idx], right[feature_idx]))
            histograms.append(hist)
            p = hist / np.sum(hist)
            probabilities.append(p)
            indices = np.digitize(feature_data, bins)
            scores.append(p[indices])
            bin_edges.append(bins)
        
        bin_edges = np.array(bin_edges)
        scores = np.array(scores).mean(axis=0)
        probabilities = np.array(probabilities)
        return bin_edges, scores, probabilities, histograms
        
    bin_edges, scores, probabilities, histograms = train(100)    
    for (bins, hist) in zip(bin_edges, histograms):
        plt.bar(bins[:-1], hist)
    plt.savefig("histograms_ex1.pdf")
    plt.clf()

    # 1.3

    def predict(bin_edges, probabilities):
        test_data = np.random.uniform(low=-3, high=3, size=(500, 2))
        test_data_projected = test_data @ proj_vectors.T

        scores = []

        for feature_idx in range(test_data_projected.shape[-1]):
            feature_data = test_data_projected[:, feature_idx]
            p = probabilities[feature_idx]
            indices = np.digitize(feature_data, bin_edges[feature_idx])
            scores.append(p[indices])

        scores = np.array(scores).mean(axis=0)
        return test_data_projected, scores

    test_data_projected, scores = predict(bin_edges, probabilities)
    plt.scatter(test_data_projected[:, 0], test_data_projected[:, 1], c=scores)
    plt.savefig("ex1.3.pdf")
    plt.clf()

    # 1.4

    for n_bins in [10, 50, 100, 150, 200]:
        bin_edges, scores, probabilities, histograms = train(n_bins)  
        test_data_projected, scores = predict(bin_edges, probabilities)
        plt.scatter(test_data_projected[:, 0], test_data_projected[:, 1], c=scores)
        plt.savefig(f"ex1.4_bins_{n_bins}.pdf".format(n_bins))
        plt.clf()
    
# ex1()