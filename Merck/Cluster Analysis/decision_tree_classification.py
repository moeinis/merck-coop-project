# Decision Tree Classification with Scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn import tree
from sklearn.model_selection import train_test_split
from yellowbrick.model_selection import FeatureImportances
import graphviz

def train_decision_tree(X, y, max_depth=5, test_size=0.1, random_state=1):
    """Train a decision tree classifier"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create and train the model
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    dt.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    
    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    
    return dt, X_train, X_test, y_train, y_test

def plot_feature_importance(model, X_train, y_train, save_path=None):
    """Plot feature importance using yellowbrick"""
    fig, ax = plt.subplots(figsize=(10, 8))
    tree_viz = FeatureImportances(model)
    tree_viz.fit(X_train, y_train)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def visualize_tree(model, feature_names, class_names=['True', 'False'], save_path=None):
    """Create tree visualization using graphviz"""
    dot_data = tree.export_graphviz(
        model,
        out_file=None,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    
    graph = graphviz.Source(dot_data)
    
    if save_path:
        png_bytes = graph.pipe(format='png')
        with open(save_path, 'wb') as f:
            f.write(png_bytes)
    
    return graph

def extract_tree_rules(tree_model, feature_names, class_names):
    """Extract human-readable rules from decision tree"""
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # Sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes), 2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def get_important_features(model, feature_names, top_n=10):
    """Get top important features from the model"""
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance_df.head(top_n)

def run_decision_tree_analysis(X, y, feature_names, max_depth=5):
    """Complete decision tree analysis pipeline"""
    print("Training Decision Tree...")
    dt_model, X_train, X_test, y_train, y_test = train_decision_tree(X, y, max_depth=max_depth)
    
    print("\nFeature Importance:")
    importance_df = get_important_features(dt_model, feature_names)
    print(importance_df)
    
    print("\nExtracting Rules...")
    rules = extract_tree_rules(dt_model, feature_names, ['True', 'False'])
    
    print("\nTop 10 Rules:")
    for i, rule in enumerate(rules[:10], 1):
        print(f"{i}. {rule}")
    
    return dt_model, rules, importance_df

if __name__ == "__main__":
    # Example usage - you would load your actual data here
    print("Decision Tree Classification Module")
    print("Use run_decision_tree_analysis() with your data")
