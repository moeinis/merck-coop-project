# PyCaret Classification Analysis
import pandas as pd
import numpy as np
from pycaret.classification import *

def setup_pycaret_experiment(df, target_column='final_label', session_id=123):
    """Setup PyCaret classification experiment"""
    exp_clf = setup(
        data=df,
        target=target_column,
        session_id=session_id,
        ignore_low_variance=True,
        combine_rare_levels=True,
        remove_multicollinearity=True,
        use_gpu=True,
        train_size=0.8
    )
    return exp_clf

def compare_classification_models():
    """Compare multiple classification models"""
    print("Comparing classification models...")
    best_models = compare_models(
        include=['dt', 'rf', 'lightgbm', 'gbc', 'ada', 'lr', 'nb', 'knn'],
        sort='Accuracy',
        n_select=3
    )
    return best_models

def create_and_tune_model(model_name='dt'):
    """Create and tune a specific model"""
    print(f"Creating {model_name} model...")
    model = create_model(model_name)
    
    print(f"Tuning {model_name} model...")
    tuned_model = tune_model(model)
    
    return model, tuned_model

def evaluate_model(model, plot_types=['auc', 'pr', 'confusion_matrix', 'feature']):
    """Evaluate model with various plots"""
    evaluation_results = {}
    
    for plot_type in plot_types:
        try:
            print(f"Creating {plot_type} plot...")
            plot_model(model, plot=plot_type)
            evaluation_results[plot_type] = "Success"
        except Exception as e:
            print(f"Error creating {plot_type} plot: {e}")
            evaluation_results[plot_type] = f"Error: {e}"
    
    return evaluation_results

def make_predictions(model):
    """Make predictions on test set"""
    print("Making predictions on test set...")
    predictions = predict_model(model)
    return predictions

def finalize_and_save_model(model, model_name="final_model"):
    """Finalize model for deployment and save"""
    print("Finalizing model...")
    final_model = finalize_model(model)
    
    print(f"Saving model as {model_name}...")
    save_model(final_model, model_name)
    
    return final_model

def run_complete_classification_pipeline(df, target_column='final_label', 
                                       model_types=['dt', 'rf', 'lightgbm']):
    """Run complete classification pipeline"""
    results = {}
    
    # Setup experiment
    print("Setting up PyCaret experiment...")
    setup_pycaret_experiment(df, target_column)
    
    # Compare models
    print("\nComparing models...")
    best_models = compare_classification_models()
    results['best_models'] = best_models
    
    # Create and tune specific models
    trained_models = {}
    for model_type in model_types:
        print(f"\nWorking with {model_type}...")
        model, tuned_model = create_and_tune_model(model_type)
        trained_models[model_type] = {'base': model, 'tuned': tuned_model}
        
        # Evaluate tuned model
        print(f"Evaluating tuned {model_type}...")
        evaluation = evaluate_model(tuned_model)
        trained_models[model_type]['evaluation'] = evaluation
        
        # Make predictions
        predictions = make_predictions(tuned_model)
        trained_models[model_type]['predictions'] = predictions
    
    results['trained_models'] = trained_models
    
    # Select best model for finalization (using first model as example)
    best_model_name = model_types[0]
    best_model = trained_models[best_model_name]['tuned']
    
    print(f"\nFinalizing {best_model_name} model...")
    final_model = finalize_and_save_model(best_model, f"final_{best_model_name}_model")
    results['final_model'] = final_model
    
    return results

def evaluate_model_performance(predictions_df, actual_col='final_label', pred_col='Label'):
    """Evaluate model performance on predictions"""
    from pycaret.utils import check_metric
    
    try:
        accuracy = check_metric(predictions_df[actual_col], predictions_df[pred_col], metric='Accuracy')
        precision = check_metric(predictions_df[actual_col], predictions_df[pred_col], metric='Precision')
        recall = check_metric(predictions_df[actual_col], predictions_df[pred_col], metric='Recall')
        f1 = check_metric(predictions_df[actual_col], predictions_df[pred_col], metric='F1')
        
        performance_metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        
        print("Performance Metrics:")
        for metric, value in performance_metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return performance_metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

if __name__ == "__main__":
    print("PyCaret Classification Module")
    print("Use run_complete_classification_pipeline() with your data")
