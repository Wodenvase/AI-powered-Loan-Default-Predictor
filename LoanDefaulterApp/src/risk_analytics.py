import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def risk_segmentation(probabilities):
    """
    Segment customers into risk categories based on default probability.
    """
    segments = []
    for prob in probabilities:
        if prob < 0.2:
            segments.append('Low Risk')
        elif prob < 0.5:
            segments.append('Medium Risk')
        else:
            segments.append('High Risk')
    return segments

def approve_reject_decision(probabilities, loan_amounts, credit_scores, loan_types, income_levels):
    """
    Determine approve/reject decisions based on prediction probability and business rules.
    
    Business Rules:
    - Reject if default probability > 0.7
    - Reject if credit score < 600 and default probability > 0.5
    - Reject if loan amount > 2x annual income and default probability > 0.4
    - Reject revolving loans if default probability > 0.6
    - Reject cash loans if default probability > 0.5
    """
    decisions = []
    
    for i, prob in enumerate(probabilities):
        loan_amount = loan_amounts[i]
        credit_score = credit_scores[i]
        loan_type = loan_types[i]
        income = income_levels[i]
        
        # Initialize as approve
        decision = 'Approve'
        
        # Rule 1: High default probability
        if prob > 0.7:
            decision = 'Reject'
        
        # Rule 2: Low credit score with moderate default risk
        elif credit_score < 600 and prob > 0.5:
            decision = 'Reject'
        
        # Rule 3: High loan-to-income ratio
        elif loan_amount > (2 * income) and prob > 0.4:
            decision = 'Reject'
        
        # Rule 4: Loan type specific rules
        elif loan_type == 'revolving_loan' and prob > 0.6:
            decision = 'Reject'
        elif loan_type == 'cash_loan' and prob > 0.5:
            decision = 'Reject'
        
        decisions.append(decision)
    
    return decisions 