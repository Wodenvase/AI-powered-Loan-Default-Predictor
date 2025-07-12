import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Loan Default Prediction Dashboard', fontsize=16, fontweight='bold')

# 1. Model Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [1.00, 1.00, 1.00, 1.00]
colors = ['#2E8B57', '#4682B4', '#CD853F', '#DC143C']

bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
ax1.set_title('Model Performance Metrics', fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_ylim(0, 1.1)
for bar, value in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

# 2. Approval vs Rejection Distribution
approval_data = ['Approved', 'Rejected']
approval_counts = [14, 5]
approval_colors = ['#32CD32', '#FF6347']

wedges, texts, autotexts = ax2.pie(approval_counts, labels=approval_data, colors=approval_colors, 
                                   autopct='%1.1f%%', startangle=90)
ax2.set_title('Loan Approval Distribution', fontweight='bold')

# 3. Risk Distribution by Loan Type
loan_types = ['Cash Loan', 'Revolving Loan']
low_risk = [8, 3]
high_risk = [2, 6]

x = np.arange(len(loan_types))
width = 0.35

ax3.bar(x - width/2, low_risk, width, label='Low Risk', color='#32CD32', alpha=0.7)
ax3.bar(x + width/2, high_risk, width, label='High Risk', color='#FF6347', alpha=0.7)

ax3.set_title('Risk Distribution by Loan Type', fontweight='bold')
ax3.set_ylabel('Number of Loans')
ax3.set_xticks(x)
ax3.set_xticklabels(loan_types)
ax3.legend()

# 4. Credit Score vs Loan Amount Scatter
np.random.seed(42)
credit_scores = np.random.normal(700, 50, 50)
loan_amounts = np.random.normal(15000, 5000, 50)
default_risk = np.random.beta(2, 5, 50)

scatter = ax4.scatter(credit_scores, loan_amounts, c=default_risk, cmap='RdYlGn_r', 
                      s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.set_xlabel('Credit Score')
ax4.set_ylabel('Loan Amount ($)')
ax4.set_title('Credit Score vs Loan Amount', fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Default Risk Probability')

# Add some text boxes for key insights
fig.text(0.02, 0.02, 'Key Insights:\n• 100% Model Accuracy\n• 73.7% Approval Rate\n• Automated Risk Assessment\n• Real-time Predictions', 
         fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('images/dashboard_preview.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Dashboard preview image generated successfully!") 