import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_model import train_and_save_model

def test_train_and_save_model():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'sample_loan_data.csv')
    model_path = os.path.join(base_dir, 'models', 'test_model.joblib')
    output_csv = os.path.join(base_dir, 'data', 'test_predictions.csv')
    model, X_test, y_test = train_and_save_model(data_path, model_path)
    # Generate predictions
    y_pred = model.predict(X_test)
    # Save test features, actual, and predicted values
    test_results = pd.DataFrame(X_test)
    test_results['actual'] = y_test
    test_results['predicted'] = y_pred
    test_results.to_csv(output_csv, index=False)
    print(f'Test predictions saved to {output_csv}')
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    # Error analysis
    errors = test_results[test_results['actual'] != test_results['predicted']]
    if not errors.empty:
        print('\nRows where model was wrong (actual != predicted):')
        print(errors)
    else:
        print('\nNo errors: all predictions were correct.')
    assert os.path.exists(model_path), 'Model file was not created.'
    print('Test passed: Model file created successfully.')

if __name__ == '__main__':
    test_train_and_save_model() 