import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import train_and_save_model
from risk_analytics import approve_reject_decision
from sklearn.model_selection import train_test_split

def test_train_and_save_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'sample_loan_data.csv')
    model_path = os.path.join(base_dir, 'models', 'test_model.joblib')
    output_csv = os.path.join(base_dir, 'data', 'test_predictions.csv')
    
    # Load original data to get feature names and original values
    original_data = pd.read_csv(data_path)
    print("Original data columns:", original_data.columns.tolist())
    
    model, X_test, y_test = train_and_save_model(data_path, model_path)
    
    # Print the actual feature names from the processed data
    print("Processed feature names:", X_test.columns.tolist())
    
    # Generate predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of default
    
    # Get original test data for business rules
    _, X_test_orig, _, _ = train_test_split(original_data.drop('default', axis=1), 
                                           original_data['default'], 
                                           test_size=0.2, random_state=42)
    
    # Create proper test results DataFrame with meaningful column names
    test_results = pd.DataFrame(X_test, columns=X_test.columns)
    test_results['actual'] = y_test.values
    test_results['predicted'] = y_pred
    test_results['default_probability'] = y_prob
    
    # Add original values for business rules
    test_results['original_loan_amount'] = X_test_orig['loan_amount'].values
    test_results['original_income'] = X_test_orig['income'].values
    test_results['original_credit_score'] = X_test_orig['credit_score'].values
    test_results['original_loan_type'] = X_test_orig['loan_type'].values
    
    # Generate approve/reject decisions
    decisions = approve_reject_decision(
        y_prob,
        test_results['original_loan_amount'].values,
        test_results['original_credit_score'].values,
        test_results['original_loan_type'].values,
        test_results['original_income'].values
    )
    
    test_results['approve_reject'] = decisions
    
    # Save test features, actual, and predicted values
    test_results.to_csv(output_csv, index=False)
    print(f'Test predictions saved to {output_csv}')
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    
    # Print approval statistics
    approved = sum(1 for d in decisions if d == 'Approve')
    rejected = sum(1 for d in decisions if d == 'Reject')
    print(f'Approved: {approved} ({approved/len(decisions)*100:.1f}%)')
    print(f'Rejected: {rejected} ({rejected/len(decisions)*100:.1f}%)')
    
    # Error analysis
    errors = test_results[test_results['actual'] != test_results['predicted']]
    if not errors.empty:
        print('\nRows where model was wrong (actual != predicted):')
        print(errors[['actual', 'predicted', 'default_probability', 'approve_reject']])
    else:
        print('\nNo errors: all predictions were correct.')
    
    assert os.path.exists(model_path), 'Model file was not created.'
    print('Test passed: Model file created successfully.')

if __name__ == '__main__':
    test_train_and_save_model() 