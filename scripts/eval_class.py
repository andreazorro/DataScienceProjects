import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef
from tqdm import tqdm

def evaluate_models(models, X_train, X_test, y_train, y_test):

    # Initialize empty lists to store the results for both training and test data
    results = []

    # Wrap the loop with tqdm for tracking
    for name, model in tqdm(models.items(), desc="Evaluating models"):
        
        # Model training
        model.fit(X_train, y_train)

        # Predictions and metrics for training data
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision, train_recall, train_f1_score, _ = precision_recall_fscore_support(y_train, y_train_pred, average='weighted', zero_division=0)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        # Predictions and metrics for test data
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision, test_recall, test_f1_score, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)
        test_mcc = matthews_corrcoef(y_test, y_test_pred)

        # Append the test results to the test list
        results.append({
            'Model': name,
            'Accuracy Train': train_accuracy,
            'Accuracy Test': test_accuracy,
            'Precision Train': train_precision,
            'Precision Test': test_precision,
            'Recall Train': train_recall,
            'Recall Test': test_recall,
            'F1 Train': train_f1_score,
            'F1 Test': test_f1_score,
            'MCC Train': train_mcc,
            'MCC Test': test_mcc
        })

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

    return results_df, markdown_content
