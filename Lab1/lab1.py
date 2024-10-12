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
def zScoreClassifier(data, d, mean=None, cov_matrix=None):
    import numpy as np
    from scipy import stats
    from scipy.spatial.distance import mahalanobis
    
    if d == 1:
        zscores = stats.zscore(data).flatten()
    else:
        zscores = []
        iv = np.linalg.inv(cov_matrix)
        for point in data:
            zscores.append(mahalanobis(point, mean, iv))
        zscores = np.array(zscores)    
    
    threshold = np.quantile(zscores, 0.1)
    preds = np.zeros(data.shape[0])
    preds[zscores <= threshold] = 1
    preds[zscores > threshold] = 0 
    
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

outlier_range = 3
outliers = np.random.uniform(low=-outlier_range, high=outlier_range, size=(n_outliers, 2))

data = np.vstack([normal_data, outliers])
labels = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])

preds = zScoreClassifier(data, d=2, mean=mean, cov_matrix=cov_matrix)
ba = getBalancedAcc(labels, preds)
print("2D Balanced Acc=", ba)


    
    