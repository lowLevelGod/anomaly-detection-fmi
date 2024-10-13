import pyod
import matplotlib.pyplot as plt
import pyod.models
import pyod.models.knn

# ex1
X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
normal_points = X_train[y_train == 0]
anomalous_points = X_train[y_train == 1] 
plt.scatter(normal_points[:, 0], normal_points[:, 1], color="blue")
plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], color="red")
plt.title("Training points")
plt.savefig("ex1.pdf")
plt.clf()

# ex2
def runModel(X_train, X_test, c):
    knn = pyod.models.knn.KNN(contamination=c)
    knn.fit(X_train)
    train_pred = knn.predict(X_train)
    test_pred = knn.predict(X_test)
    train_proba = knn.decision_function(X_train)
    test_proba = knn.decision_function(X_test)
    return train_pred, test_pred, train_proba, test_proba

def getBalancedAcc(y, y_pred):
    from sklearn.metrics import confusion_matrix 
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2

def plotRoc(y, y_score, c, is_train=False):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, y_score)
    roc_auc = auc(fpr, tpr)
    scope='Train' if is_train else 'Test'
    plt.title(f'{scope} ROC for c={c}'.format(c=c, scope=scope))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ex2_" + str(c) + "_" + scope + ".pdf")
    plt.clf()

for c in [0.1, 0.2, 0.3, 0.4, 0.5]:
    train_pred, test_pred, train_proba, test_proba = runModel(X_train, X_test, c)
    ba_train = getBalancedAcc(train_pred, y_train)
    ba_test = getBalancedAcc(test_pred, y_test)
    
    print("Contamination=", c)
    print("Balanced Acc for train=", ba_train)
    print("Balanced Acc for test=", ba_test)
    
    plotRoc(y_train, train_proba, c, is_train=True)
    plotRoc(y_test, test_proba, c, is_train=False)
    
# ex3
data, _, labels, _ = pyod.utils.data.generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)
def zScoreClassifier(data, d, mean=None, iv=None):
    import numpy as np
    
    if d == 1:
        zscores = np.abs(data - np.mean(data)) / np.std(data)
        zscores = zscores.flatten()
    else:
        zscores = []
        for point in data:
            zscores.append(np.sqrt((point - mean).T @ iv @ (point - mean)))
        zscores = np.array(zscores)    
    
    threshold = np.quantile(zscores, 1 - 0.1)
    preds = np.zeros(data.shape[0])
    preds[zscores <= threshold] = 0
    preds[zscores > threshold] = 1 
    
    return preds 

preds = zScoreClassifier(data, d=1)
ba = getBalancedAcc(labels, preds)
print("1D Balanced Acc=", ba)

# ex4
import numpy as np

np.random.seed(10)

mean = [0, 1]  
cov_matrix = np.array([
    [1, 0.3], 
    [0.3, 2]
])

n_samples = 1000  
contamination = 0.10  
n_outliers = int(n_samples * contamination)
n_normal = n_samples - n_outliers

normal_data = np.random.multivariate_normal(mean, cov_matrix, n_normal)

outlier_range = 5
outliers = np.random.multivariate_normal(mean, cov_matrix, n_outliers) + np.random.uniform(low=-outlier_range, high=outlier_range, size=(n_outliers, 2))

data = np.vstack([normal_data, outliers])
labels = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

def cholesky(A):
    n = len(A)

    L = [[0.0] * n for _ in range(0, n, 1)]

    for i in range(0, n, 1):
        for j in range(0, i + 1, 1):
            tmp_sum = 0
            for k in range(0, j, 1):
                tmp_sum += L[i][k] * L[j][k]
            if i == j:  
                L[i][j] = (A[i][i] - tmp_sum) ** (1 / 2)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - tmp_sum))
    return np.array(L)

def substitution(R, S):
    n = len(R)
    X = [[0.0] * n for i in range(0, n, 1)]

    for i in range(n - 1, -1, -1):
        for j in range(i, -1, -1):
            tmp_sum = 0
            for k in range(j + 1, n, 1):
                tmp_sum += R[j][k] * X[k][i]
            X[j][i] = (S[j][i] - tmp_sum) / R[j][j]
            X[i][j] = X[j][i]
    return X

def inv(A):
    L = cholesky(A)

    R = L.T

    S = np.diag(1 / R.diagonal())

    X = substitution(R, S)
    
    return X

iv = inv(cov_matrix)
preds = zScoreClassifier(data, d=2, mean=mean, iv=iv)
ba = getBalancedAcc(labels, preds)
print("2D Balanced Acc=", ba)


    
    