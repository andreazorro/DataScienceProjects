from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

def ml_reg(rs):
    # Linear Regression
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
    linear = LinearRegression()
    linear_grid_search = GridSearchCV(estimator=linear, param_grid=param_grid, n_jobs=-1, verbose=1)

    # Ridge Regression
    param_grid = {  
        'alpha': np.linspace(0.000001, 0.1, 5),
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
    ridge = Ridge(random_state=rs)
    ridge_grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, n_jobs=-1, verbose=1)

    # Lasso Regression
    param_grid = {
        'alpha': np.linspace(0.000001, 0.1, 5),
        'fit_intercept': [True, False],
        'copy_X': [True, False]
    }
    lasso = Lasso(random_state=rs)
    lasso_grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, n_jobs=-1, verbose=1)

    # LassoLars Regression
    param_grid = {
        'alpha': np.linspace(0.000001, 0.1, 5),
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    lassolars = LassoLars(random_state=rs)
    lassolars_grid_search = GridSearchCV(estimator=lassolars, param_grid=param_grid, n_jobs=-1, verbose=1)

    # ElasticNet Regression
    param_grid = {
        'alpha': np.linspace(0.000001, 0.1, 5),
        'l1_ratio': np.linspace(0.000001, 1, 5),
        'fit_intercept': [True, False],
        'positive': [True, False]
    }
    elasticnet = ElasticNet(random_state=rs)
    elasticnet_grid_search = GridSearchCV(estimator=elasticnet, param_grid=param_grid, n_jobs=-1, verbose=1)

    # Random Forest Regression
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    random_forest = RandomForestRegressor(random_state=rs)
    random_forest_grid_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_grid, n_jobs=-1, verbose=1)

    # MLP Regression
    param_grid = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': np.logspace(-1, 1, 5),
        'learning_rate': ['constant','adaptive'],
    }
    mlp = MLPRegressor(random_state=rs)
    mlp_grid_search = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid, n_jobs=-1, verbose=1)

    # Models dictionary
    models = {
        'Linear Regression': linear_grid_search,
        'Ridge Regression': ridge_grid_search,
        'Lasso Regression': lasso_grid_search,
        'LassoLars Regression': lassolars_grid_search,
        'ElasticNet Regression': elasticnet_grid_search,
        'Random Forest Regression': random_forest_grid_search, 
        'MLP Regression': mlp_grid_search
    }

    return models