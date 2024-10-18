import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from tqdm import tqdm

def evaluate_models(models, X_train, X_test, y_train, y_test):

    # Initialize empty lists to store the results for both training and test data
    results = []
    parameters = {}

    # Wrap the loop with tqdm for tracking
    for name, model in tqdm(models.items(), desc="Evaluating models"):
        
        # Model training
        model.fit(X_train, y_train)
        model_params = model.best_estimator_ if hasattr(model, 'best_estimator_') else model

        # Predictions and metrics for training data
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Predictions and metrics for test data
        y_test_pred = model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Append the test results to the test list
        results.append({
            'Model': name,
            'MAE Train': train_mae,
            'MAE Test': test_mae,
            'RMSE Train': train_rmse,
            'RMSE Test': test_rmse,
            'R2 Train': train_r2,
            'R2 Test': test_r2
        })

        # Append the parameters to the dictionary

        parameters[name] = model_params

    # Convert the list to a DataFrame
    results_df = pd.DataFrame(results)

    # Generate Markdown content for the table
    markdown_content = "### Model Evaluation\n\n"

    # Convert DataFrame to Markdown
    if not results_df.empty:
        table_md = results_df.to_markdown(index=False)
        # Add title for the metrics table
        markdown_content += "#### Metrics\n\n"
        
        # Center the headers for the table
        lines = table_md.split('\n')
        if len(lines) > 1:
            header = lines[0]
            separator = lines[1]
            centered_header = '| ' + ' | '.join([f'<center>{col}</center>' for col in header.split('|')[1:-1]]) + ' |'
            centered_table_md = '\n'.join([centered_header, separator] + lines[2:])
            # Add the centered table to the Markdown content
            markdown_content += centered_table_md + "\n\n"

    return parameters, markdown_content