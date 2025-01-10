import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, f1_score, 
                           accuracy_score, precision_score, recall_score)
import pandas as pd
import psutil
import sys
import traceback

def print_memory_usage():
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB", flush=True)

def load_embeddings(directory):
    """Load all embeddings from a directory."""
    print(f"\nLoading embeddings from {directory}", flush=True)
    embeddings = []
    total_files = 0
    skipped_files = 0
    
    for root, _, files in os.walk(directory):
        for file in sorted(files):
            if file.endswith('_embeddings.npy'):
                total_files += 1
                path = os.path.join(root, file)
                try:
                    emb = np.load(path)
                    if emb.size == 0 or (len(emb.shape) == 1 and emb.shape[0] != 768):
                        skipped_files += 1
                        continue
                    if len(emb.shape) == 1:
                        emb = emb.reshape(1, -1)
                    embeddings.append(emb)
                except Exception as e:
                    skipped_files += 1
                    continue
    
    if not embeddings:
        return np.array([])
    
    result = np.concatenate(embeddings, axis=0)
    print(f"Loaded embeddings shape: {result.shape}", flush=True)
    return result

def load_test_embeddings_by_length(directory):
    """Load test embeddings for each length group."""
    length_groups = ['less_than_500bp', '500_to_1000bp', '1000_to_2000bp']
    embeddings_by_length = {}
    
    for group in length_groups:
        group_dir = os.path.join(directory, group)
        if os.path.exists(group_dir):
            embeddings = load_embeddings(group_dir)
            if len(embeddings) > 0:
                embeddings_by_length[group] = embeddings
                print(f"{group} embeddings shape: {embeddings_by_length[group].shape}", flush=True)
    
    return embeddings_by_length

def train_and_evaluate(X_train, y_train, test_data, results_dir='./results'):
    """Train Random Forest classifier and evaluate on different length groups."""
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, 'length_results.csv')
    
    # Grid search parameters from Feng et al. 2024
    param_grid = {
        'n_estimators': [1000, 500, 200],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [20, None],
        'min_samples_split': [2, 5]
    }
    
    # Initialize and train model
    rf = RandomForestClassifier(random_state=42, verbose=0, n_jobs=40)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=4,  # 4-fold cross-validation as in Feng et al. 2024
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nStarting grid search...", flush=True)
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}", flush=True)
    
    # Evaluate on each length group
    results = []
    for group, (X_test, y_test) in test_data.items():
        # Make predictions
        test_pred = best_model.predict(X_test)
        test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'length_group': group,
            'mcc': matthews_corrcoef(y_test, test_pred),
            'auc': roc_auc_score(y_test, test_pred_proba),
            'f1': f1_score(y_test, test_pred),
            'accuracy': accuracy_score(y_test, test_pred),
            'precision': precision_score(y_test, test_pred),
            'recall': recall_score(y_test, test_pred),
            'n_samples': len(y_test)
        }
        results.append(metrics)
        
        # Save predictions for ROC curves
        preds_df = pd.DataFrame({'True': y_test, 'Pred': test_pred_proba})
        pred_path = os.path.join(results_dir, f'predictions_{group}.csv')
        preds_df.to_csv(pred_path, index=False)
        
        print(f"\nMetrics for {group}:", flush=True)
        for metric, value in metrics.items():
            if metric != 'length_group':
                print(f"{metric}: {value:.4f}", flush=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}", flush=True)
    
    return best_model, results_df

if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)
    
    try:
        # Load embeddings
        embeddings_dir = "./embeddings"
        train_dir = os.path.join(embeddings_dir, "train")
        test_dir = os.path.join(embeddings_dir, "test")
        
        # Load training data
        print("Loading training embeddings...", flush=True)
        viral_train = load_embeddings(os.path.join(train_dir, "viral"))
        bacterial_train = load_embeddings(os.path.join(train_dir, "bacterial"))
        
        # Create training dataset
        X_train = np.concatenate([viral_train, bacterial_train], axis=0)
        y_train = np.concatenate([np.ones(len(viral_train)), np.zeros(len(bacterial_train))])
        
        # Load test data by length groups
        print("\nLoading test embeddings...", flush=True)
        viral_test = load_test_embeddings_by_length(os.path.join(test_dir, "viral"))
        bacterial_test = load_test_embeddings_by_length(os.path.join(test_dir, "bacterial"))
        
        # Create test datasets by length
        test_data = {}
        for group in viral_test.keys():
            if group in bacterial_test:
                X_test = np.concatenate([viral_test[group], bacterial_test[group]], axis=0)
                y_test = np.concatenate([
                    np.ones(len(viral_test[group])), 
                    np.zeros(len(bacterial_test[group]))
                ])
                test_data[group] = (X_test, y_test)
        
        # Train and evaluate model
        model, results = train_and_evaluate(X_train, y_train, test_data)
        
    except Exception as e:
        print("\nError in execution:", flush=True)
        print(traceback.format_exc(), flush=True)
        sys.exit(1)