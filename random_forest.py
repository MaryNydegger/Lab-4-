import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from sklearn.metrics import accuracy_score, r2_score

def random_forest(X, y, ntrees, max_depth, classification = True):
    n_samples, n_features = X.shape

    trees = []
    importances = np.zeros(n_features)
    ypred = np.zeros((n_samples, ntrees))

    oob_samples = [[] for _ in range(n_samples)]
    oob_predictions = [[] for _ in range(n_samples)]

    for _ in range(ntrees):
        indices = np.random.choice(n_samples, n_samples, replace = True)
        X_boot = X[indices]
        y_boot = y[indices]

        subset_indices = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        X_subset = X_boot[:, subset_indices]

        if classification:
            tree = DecisionTreeClassifier(max_depth = max_depth)
            tree.fit(X_subset, y_boot)
            y_pred = tree.predict(X_subset)
            importances[subset_indices] += tree.feature_importances_
        else:
            tree = DecisionTreeRegressor(max_depth = max_depth)
            tree.fit(X_subset, y_boot)
            y_pred = tree.predict(X_subset)
            importances[subset_indices] += tree.feature_importances_

        trees.append(tree)

        for i in range(n_samples):
            if i not in indices:
                oob_samples[i].append(y_pred[i])

    oob = 0 
    for i in range(n_samples):
        if len(oob_samples[i]) > 0:
            oob_predictions[i] = np.mean(oob_samples[i])
            if classification: 
                oob += accuracy_score([y[i]], [int(round(oob_predictions[i]))])
            else:
                oob += r2_score([y[i]], [oob_predictions[i]])

    oob /= n_samples

    return trees, importances, oob, ypred



# X = np.random.rand(100, 5)
# y = np.random.randint(0, 2, 100)
# ntrees = 150
# max_depth = 5
# trees, importances, oob, ypred = random_forest(X, y, ntrees, max_depth, classification=True)