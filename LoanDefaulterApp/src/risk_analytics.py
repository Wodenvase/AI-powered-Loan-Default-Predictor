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

def risk_segmentation(probabilities, thresholds=(0.3, 0.7)):
    """
    Segment risk into 'Low', 'Medium', 'High' based on probability thresholds.
    """
    segments = []
    for p in probabilities:
        if p < thresholds[0]:
            segments.append('Low')
        elif p < thresholds[1]:
            segments.append('Medium')
        else:
            segments.append('High')
    return segments 