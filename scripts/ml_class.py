from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import numpy as np

def ml_class_im(cv, rs):

    # Define a custom scorer for multiclass classification
    scorer = make_scorer(f1_score, average='weighted')

    # Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],  # Removed 'lbfgs' due to incompatibility with 'l1'
        'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    }
    logistic_model = LogisticRegression(random_state=rs)
    logistic_grid = RandomizedSearchCV(logistic_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 19, 21],  # Add higher values
        'weights': ['uniform', 'distance'],  # 'distance' can reduce overfitting
        'metric': ['manhattan', 'minkowski'],  # Use selective metrics
        'p': [1, 2],  # 1 for Manhattan, 2 for Euclidean
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    knn_model = KNeighborsClassifier(weights='distance')
    knn_grid = RandomizedSearchCV(knn_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Gaussian Naive Bayes
    param_grid = {
        'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    }
    gnb_model = GaussianNB()
    gnb_grid = RandomizedSearchCV(gnb_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Decision Tree
    param_grid = {
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': np.linspace(0, 0.05, 50),
        'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    }
    dt_model = DecisionTreeClassifier(random_state=rs)
    dt_grid = RandomizedSearchCV(dt_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],  # Drop 'auto' to reduce complexity
        'bootstrap': [True],  # Generally bootstrap=True is better for reducing overfitting
        'ccp_alpha': np.linspace(0, 0.05, 50),
        'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    }
    rf_model = RandomForestClassifier(random_state=rs)
    rf_grid = RandomizedSearchCV(rf_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Extra Trees
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [False],  # Extra Trees generally performs better without bootstrapping
        'ccp_alpha': np.linspace(0, 0.05, 50),
        'class_weight': [{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}]
    }
    et_model = ExtraTreesClassifier(random_state=rs)
    et_grid = RandomizedSearchCV(et_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1],  # Use lower learning rates to avoid overfitting
    }
    ab_model = AdaBoostClassifier(random_state=rs, algorithm='SAMME')
    ab_grid = RandomizedSearchCV(ab_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Neural Networks
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100), (100, 50, 100)],
        'activation': ['tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
    mlp_model = MLPClassifier(random_state=rs, max_iter=5000, solver='adam', early_stopping=True)
    mlp_grid = RandomizedSearchCV(mlp_model, param_grid, cv=cv, n_jobs=-1, scoring=scorer)

    # Models dictionary 
    models = {
        'Logistic Regression': logistic_grid,
        'KNN': knn_grid,
        #'SVM': svm_grid,
        'Gaussian Naive Bayes': gnb_grid,
        'Decision Tree': dt_grid,
        'Random Forest': rf_grid,
        'Extra Trees': et_grid,
        'AdaBoost': ab_grid,
        'Neural Networks': mlp_grid
    }

    return models

def ml_class(cv, rs):
    # Logistic Regression
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],  # Removed 'lbfgs' due to incompatibility with 'l1'
    }
    logistic_model = LogisticRegression(random_state=rs)
    logistic_grid = RandomizedSearchCV(logistic_model, param_grid, cv=cv, n_jobs=-1)

    # KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 19, 21],  # Add higher values
        'weights': ['uniform', 'distance'],  # 'distance' can reduce overfitting
        'metric': ['manhattan', 'minkowski'],  # Use selective metrics
        'p': [1, 2],  # 1 for Manhattan, 2 for Euclidean
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }
    knn_model = KNeighborsClassifier(weights='distance')
    knn_grid = RandomizedSearchCV(knn_model, param_grid, cv=cv, n_jobs=-1)

    # Gaussian Naive Bayes
    param_grid = {
        'var_smoothing': [1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
    }
    gnb_model = GaussianNB()
    gnb_grid = RandomizedSearchCV(gnb_model, param_grid, cv=cv, n_jobs=-1)

    # Decision Tree
    param_grid = {
        'max_depth': [2, 4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'ccp_alpha': np.linspace(0, 0.05, 50),
    }
    dt_model = DecisionTreeClassifier(random_state=rs)
    dt_grid = RandomizedSearchCV(dt_model, param_grid, cv=cv, n_jobs=-1)

    # Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],  # Drop 'auto' to reduce complexity
        'bootstrap': [True],  # Generally bootstrap=True is better for reducing overfitting
        'ccp_alpha': np.linspace(0, 0.05, 50),
    }
    rf_model = RandomForestClassifier(random_state=rs)
    rf_grid = RandomizedSearchCV(rf_model, param_grid, cv=cv, n_jobs=-1)

    # Extra Trees
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [False],  # Extra Trees generally performs better without bootstrapping
        'ccp_alpha': np.linspace(0, 0.05, 50),
    }
    et_model = ExtraTreesClassifier(random_state=rs)
    et_grid = RandomizedSearchCV(et_model, param_grid, cv=cv, n_jobs=-1)

    # AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1],  # Use lower learning rates to avoid overfitting
    }
    ab_model = AdaBoostClassifier(random_state=rs, algorithm='SAMME')
    ab_grid = RandomizedSearchCV(ab_model, param_grid, cv=cv, n_jobs=-1)

    # Neural Networks
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 50), (100, 100), (100, 50, 100)],
        'activation': ['tanh', 'relu'],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }
    mlp_model = MLPClassifier(random_state=rs, max_iter=5000, solver='adam', early_stopping=True)
    mlp_grid = RandomizedSearchCV(mlp_model, param_grid, cv=cv, n_jobs=-1)

    # Models dictionary 
    models = {
        'Logistic Regression': logistic_grid,
        'KNN': knn_grid,
        #'SVM': svm_grid,
        'Gaussian Naive Bayes': gnb_grid,
        'Decision Tree': dt_grid,
        'Random Forest': rf_grid,
        'Extra Trees': et_grid,
        'AdaBoost': ab_grid,
        'Neural Networks': mlp_grid
    }

    return models