import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import os

def create_plots(results_dir='./results'):
    # Read the results
    results = pd.read_csv(os.path.join(results_dir, 'length_results.csv'))
    
    # Set colors
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
    
    # Create figure for bar plots
    fig1, axes = plt.subplots(2, 3, figsize=(20, 12))  # Changed to 2x3
    
    # Metrics to plot
    metrics = ['auc', 'mcc', 'f1', 'accuracy', 'precision', 'recall']
    titles = ['AUC Score', 'Matthews Correlation Coefficient', 'F1 Score', 
             'Accuracy', 'Precision', 'Recall']
    ylabels = ['AUC', 'MCC', 'F1', 'Accuracy', 'Precision', 'Recall']
    
    # Create each subplot for metrics
    for ax, metric, title, ylabel in zip(axes.flat, metrics, titles, ylabels):
        # Create bar plot
        bars = ax.bar(
            range(len(results)),
            results[metric],
            color=colors[:len(results)]
        )
        
        # Customize subplot
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('Sequence Length', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Set x-ticks
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(results['length_group'], rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=10
            )
        
        # Set y-axis limits from 0 to 1
        ax.set_ylim(0, 1)
        
        # Add gridlines
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout for bar plots
    plt.tight_layout()
    
    # Add a title for the bar plots
    fig1.suptitle('Performance Metrics by Sequence Length', y=1.02, fontsize=16)
    
    # Save bar plots
    plt.savefig(os.path.join(results_dir, 'metrics_by_length.png'), dpi=300, bbox_inches='tight')
    
    # Create ROC curves
    fig2 = plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each length group
    for i, group in enumerate(results['length_group']):
        # Load predictions for this group
        preds = pd.read_csv(os.path.join(results_dir, f'predictions_{group}.csv'))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(preds['True'], preds['Pred'])
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr,
            label=f'{group} (AUC = {roc_auc:.3f})',
            color=colors[i],
            linewidth=2
        )
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Customize ROC plot
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Sequence Length', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save ROC plot
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    
    # Print numerical results
    print("\nNumerical Results:")
    print(results.to_string(index=False))
    
    # Print number of samples in each length group
    print("\nNumber of samples in each length group:")
    print(results[['length_group', 'n_samples']].to_string(index=False))

if __name__ == "__main__":
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    create_plots(results_dir)