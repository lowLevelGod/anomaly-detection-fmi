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
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    S_inv = np.diag(1 / S)
    
    X_pseudo_inv = Vt.T @ S_inv @ U.T
    
    H = X @ X_pseudo_inv
    
    leverage = np.diag(H)
    
    return leverage

import matplotlib.pyplot as plt       

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
                     





    