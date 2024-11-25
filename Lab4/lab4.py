# ex 1.1

def ex1():
    import pyod
    X_train, X_test, y_train, y_test = pyod.utils.data.generate_data(n_train=300, n_test=200, n_features=3, contamination=0.15)
    # ex 1.2
    from pyod.models.ocsvm import OCSVM
    ocsvm = OCSVM(kernel='linear', contamination=0.15)
    ocsvm.fit(X_train)
    test_pred = ocsvm.predict(X_test)
    test_scores = ocsvm.decision_function(X_test)

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    bacc = balanced_accuracy_score(y_test, test_pred)
    roc_auc_score = roc_auc_score(y_test, test_scores)
    print("OCSVM linear -> ", "bacc=", bacc, "AUC=", roc_auc_score)

    # ex 1.3
    train_pred = ocsvm.predict(X_train)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(20, 8), subplot_kw=dict(projection="3d"))
    ax[0][0].scatter(*X_train[y_train == 0].T, c='blue')
    ax[0][0].scatter(*X_train[y_train == 1].T, c='red')
    ax[0][0].set_title("Train ground truth")
    ax[0][1].scatter(*X_test[y_test == 0].T, c='blue')
    ax[0][1].scatter(*X_test[y_test == 1].T, c='red')
    ax[0][1].set_title("Test ground truth")
    ax[1][0].scatter(*X_train[train_pred == 0].T, c='blue')
    ax[1][0].scatter(*X_train[train_pred == 1].T, c='red')
    ax[1][0].set_title("Train predicted")
    ax[1][1].scatter(*X_test[test_pred == 0].T, c='blue')
    ax[1][1].scatter(*X_test[test_pred == 1].T, c='red')
    ax[1][1].set_title("Test predicted")
    fig.savefig("ex1.3.pdf")
    plt.clf()

    # ex 1.4
    def train_and_plot_ocsvm(model, X_train, X_test, y_train, y_test, file_name, model_name):
        model.fit(X_train)
        train_pred = ocsvm.predict(X_train)
        test_pred = model.predict(X_test)
        test_scores = model.decision_function(X_test)
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        bacc = balanced_accuracy_score(y_test, test_pred)
        roc_auc_score = roc_auc_score(y_test, test_scores)
        print("OCSVM ", model_name," -> ", "bacc=", bacc, "AUC=", roc_auc_score)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(20, 8), subplot_kw=dict(projection="3d"))
        ax[0][0].scatter(*X_train[y_train == 0].T, c='blue')
        ax[0][0].scatter(*X_train[y_train == 1].T, c='red')
        ax[0][0].set_title("Train ground truth")
        ax[0][1].scatter(*X_test[y_test == 0].T, c='blue')
        ax[0][1].scatter(*X_test[y_test == 1].T, c='red')
        ax[0][1].set_title("Test ground truth")
        ax[1][0].scatter(*X_train[train_pred == 0].T, c='blue')
        ax[1][0].scatter(*X_train[train_pred == 1].T, c='red')
        ax[1][0].set_title("Train predicted")
        ax[1][1].scatter(*X_test[test_pred == 0].T, c='blue')
        ax[1][1].scatter(*X_test[test_pred == 1].T, c='red')
        ax[1][1].set_title("Test predicted")
        fig.savefig(file_name)
        fig.suptitle(model_name)
        # plt.show()
        plt.clf()
        
    for contamination in [0.15, 0.20, 0.30, 0.40, 0.45]:
        train_and_plot_ocsvm(OCSVM(kernel='linear', contamination=contamination), X_train, X_test, y_train, y_test, "ex1.3c=" + str(contamination) + ".pdf", "linear c=" + str(contamination))

    
    for contamination in [0.15, 0.20, 0.30, 0.40, 0.45]:
        train_and_plot_ocsvm(OCSVM(kernel='rbf', contamination=contamination), X_train, X_test, y_train, y_test, "ex1.4c=" + str(contamination) + ".pdf", "rbf c=" + str(contamination))

    # ex 1.5
    from pyod.models.deep_svdd import DeepSVDD
    for contamination in [0.15, 0.20, 0.30, 0.40, 0.45]:
        deepSvdd = DeepSVDD(n_features=3, contamination=0.15)
        train_and_plot_ocsvm(deepSvdd, X_train, X_test, y_train, y_test, "ex1.5c=" + str(contamination) + ".pdf", "DeepSVDD c=" + str(contamination))

def ex2():
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    mat = loadmat("./cardio.mat")
    data, labels = mat['X'], mat['y']
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.6, random_state=42)
    from sklearn.svm import OneClassSVM
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    contamination = (train_labels.sum()) / train_labels.shape[0]
    base_estimator = Pipeline([("preprocess", StandardScaler()), ("ocsvm", OneClassSVM())])
    import numpy as np
    nu = np.linspace(start=0.1, stop=0.9, num=10)
    nu = np.append(nu, [contamination])
    gamma = np.array([0.01, 0.1, 0.5, 1, 2, 5])
    param_grid = [
        {"ocsvm__kernel": ["linear"], "ocsvm__nu": nu},
        {"ocsvm__kernel": ["rbf"], "ocsvm__gamma": ["scale", "auto", *gamma], "ocsvm__nu": nu},
    ]
    
    train_labels = -(train_labels * 2 - 1)
    test_labels = -(test_labels * 2 - 1)
    from sklearn.model_selection import GridSearchCV
    search = GridSearchCV(estimator=base_estimator, param_grid=param_grid, scoring="balanced_accuracy")
    model = search.fit(train_data, train_labels)
    train_bacc = model.score(train_data, train_labels)
    test_bacc = model.score(test_data, test_labels)
    print("Train bacc=", train_bacc)
    print("Test bacc=", test_bacc)
    
    for k, v in model.best_params_.items():
        print(k, "=", v)
 
    best_model = OneClassSVM(
        kernel=model.best_params_.get("ocsvm__kernel"),
        gamma=model.best_params_.get("ocsvm__gamma"),
        nu=model.best_params_.get("ocsvm__nu"),
    )
    best_model.fit(train_data, train_labels)
    train_pred = best_model.predict(train_data)
    test_pred = best_model.predict(test_data)
    train_scores = best_model.decision_function(train_data)
    test_scores = best_model.decision_function(test_data)
    
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    train_bacc = balanced_accuracy_score(train_labels, train_pred)
    train_bacc = balanced_accuracy_score(train_labels, train_pred)
    test_bacc = balanced_accuracy_score(test_labels, test_pred)
    train_auc = roc_auc_score(train_labels, train_scores)
    test_auc = roc_auc_score(test_labels, test_scores)
    
    print("Train bacc=", train_bacc)
    print("Test bacc=", test_bacc)
    print("Train auc=", train_auc)
    print("Test auc=", test_auc)

def ex3():
    from scipy.io import loadmat
    from sklearn.model_selection import train_test_split
    mat = loadmat("./shuttle.mat")
    data, labels = mat['X'], mat['y']
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5, random_state=42)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)
    
    from pyod.models.ocsvm import OCSVM
    from pyod.models.deep_svdd import DeepSVDD
    
    def train_and_score(model, train_data, test_data, train_labels, test_labels):
        model.fit(train_data)
        test_pred = model.predict(test_data)
        test_scores = model.decision_function(test_data)
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        bacc = balanced_accuracy_score(test_labels, test_pred)
        roc_auc_score = roc_auc_score(test_labels, test_scores)
        
        print("Bacc=", bacc)
        print("ROC=", roc_auc_score)
    
    contamination = (train_labels.sum()) / train_labels.shape[0]
    train_and_score(OCSVM(contamination=contamination), train_data, test_data, train_labels, test_labels)
    train_and_score(DeepSVDD(n_features=train_data.shape[-1], contamination=contamination, hidden_neurons=[32, 16]), train_data, test_data, train_labels, test_labels)
    train_and_score(DeepSVDD(n_features=train_data.shape[-1], contamination=contamination, hidden_neurons=[64, 32, 16]), train_data, test_data, train_labels, test_labels)
    train_and_score(DeepSVDD(n_features=train_data.shape[-1], contamination=contamination, hidden_neurons=[64, 64, 16]), train_data, test_data, train_labels, test_labels)
        
    
ex1()
# ex2()
# ex3()