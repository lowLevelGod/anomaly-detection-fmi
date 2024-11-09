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
    
# ex2
def ex2():
    # 2.1
    import numpy as np
    train_data, _ = make_blobs(n_samples=500, n_features=2, centers=[[10, 0], [0, 10]])
    from pyod.models.iforest import IForest
    from pyod.models.dif import DIF
    from pyod.models.loda import LODA
    
    # 2.2
    iforest = IForest(contamination=0.02)
    dif = DIF(contamination=0.02)
    loda = LODA(contamination=0.02)
    
    # 2.3 + 2.4
    iforest.fit(train_data)
    dif.fit(train_data)
    loda.fit(train_data)
    test_data = np.random.uniform(low=-10, high=20, size=(1000, 2))
    scores_iforest = iforest.decision_function(test_data)
    scores_dif = dif.decision_function(test_data)
    scores_loda = loda.decision_function(test_data)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    ax[0].scatter(test_data[:, 0], test_data[:, 1], c=scores_iforest)
    ax[0].set_title("IForest")
    ax[1].scatter(test_data[:, 0], test_data[:, 1], c=scores_dif)
    ax[1].set_title("DIF")
    ax[2].scatter(test_data[:, 0], test_data[:, 1], c=scores_loda)
    ax[2].set_title("LODA")
    fig.savefig("ex2.1.pdf")
    plt.clf()

# ex3
def ex3():
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    # 3.1
    mat = loadmat("./shuttle.mat")
    data, labels = mat['X'], mat['y']
    
    # 3.2
    from pyod.models.iforest import IForest
    from pyod.models.dif import DIF
    from pyod.models.loda import LODA
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    
    mean_ba_iforest = 0
    mean_ba_dif = 0
    mean_ba_loda = 0
    mean_roc_iforest = 0
    mean_roc_dif = 0
    mean_roc_loda = 0
    cnt = 1
    for seed in range(10):
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.4, random_state=seed)
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        
        c = np.sum(train_labels) / train_labels.shape[0]
        iforest = IForest(contamination=c)
        dif = DIF(contamination=c)
        loda = LODA(contamination=c)
        iforest.fit(train_data)
        dif.fit(train_data)
        loda.fit(train_data)
        
        iforest_preds = iforest.predict(test_data)
        iforest_scores = iforest.decision_function(test_data)
        dif_preds = dif.predict(test_data)
        dif_scores = dif.decision_function(test_data)
        loda_preds = loda.predict(test_data)
        loda_scores = loda.decision_function(test_data)
        
        ba_iforest = balanced_accuracy_score(test_labels, iforest_preds)
        ba_dif = balanced_accuracy_score(test_labels, dif_preds)
        ba_loda = balanced_accuracy_score(test_labels, loda_preds)
        
        roc_iforest = roc_auc_score(test_labels, iforest_scores)
        roc_dif = roc_auc_score(test_labels, dif_scores)
        roc_loda = roc_auc_score(test_labels, loda_scores)
        
        print("BA IForest=", ba_iforest, "ROC IForest=", roc_iforest)
        print("BA DIF=", ba_dif , "ROC DIF=", roc_dif)
        print("BA LODA=", ba_loda, "ROC LODA=", roc_loda)
        
        mean_ba_iforest += (ba_iforest - mean_ba_iforest) / cnt
        mean_ba_dif += (ba_dif - mean_ba_dif) / cnt
        mean_ba_loda += (ba_loda - mean_ba_loda) / cnt
        mean_roc_iforest += (roc_iforest - mean_roc_iforest) / cnt
        mean_roc_dif += (roc_dif - mean_roc_dif) / cnt
        mean_roc_loda += (roc_loda - mean_roc_loda) / cnt
        
        cnt += 1
    print("MEAN BA IForest=", mean_ba_iforest, "MEAN ROC IForest=", mean_roc_iforest)
    print("MEAN BA DIF=", mean_ba_dif , "MEAN ROC DIF=", mean_roc_dif)
    print("MEAN BA LODA=", mean_ba_loda, "MEAN ROC LODA=", mean_roc_loda)

# ex1()
# ex2()
ex3()