"""
Decision Tree Models Implementation

This module implements three decision tree algorithms as used in the paper:
1. J48 (C4.5) - Using sklearn DecisionTreeClassifier with entropy criterion
2. Random Tree - Using sklearn DecisionTreeClassifier with random splits
3. REPTree - Using sklearn DecisionTreeClassifier with reduced error pruning

The paper uses 10-fold cross-validation for evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import pickle


class J48Classifier:
    """
    J48 (C4.5) Decision Tree Classifier.

    This is implemented using sklearn's DecisionTreeClassifier with:
    - criterion='entropy' (information gain as in C4.5)
    - splitter='best' (best split at each node)
    - Pruning through min_samples_split and min_samples_leaf
    """

    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=None):
        """
        Initialize J48 classifier.

        Parameters:
        -----------
        min_samples_split : int
            Minimum samples required to split an internal node
        min_samples_leaf : int
            Minimum samples required to be at a leaf node
        max_depth : int, optional
            Maximum depth of the tree
        """
        self.model = DecisionTreeClassifier(
            criterion='entropy',  # Information gain (like C4.5/J48)
            splitter='best',      # Best split at each node
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=42
        )
        self.name = "J48"

    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def get_model(self):
        """Get the underlying sklearn model"""
        return self.model


class RandomTreeClassifier:
    """
    Random Tree Classifier.

    Implementation using sklearn's DecisionTreeClassifier with:
    - splitter='random' (random splits)
    - No pruning (max_features can be used to limit features considered)
    """

    def __init__(self, max_features='sqrt', max_depth=None):
        """
        Initialize Random Tree classifier.

        Parameters:
        -----------
        max_features : str or int
            Number of features to consider for best split
        max_depth : int, optional
            Maximum depth of the tree
        """
        self.model = DecisionTreeClassifier(
            criterion='gini',     # Gini impurity
            splitter='random',    # Random splits
            max_features=max_features,
            max_depth=max_depth,
            random_state=42
        )
        self.name = "RandomTree"

    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def get_model(self):
        """Get the underlying sklearn model"""
        return self.model


class REPTreeClassifier:
    """
    REPTree (Reduced Error Pruning Tree) Classifier.

    Implementation using sklearn's DecisionTreeClassifier with:
    - criterion='entropy'
    - ccp_alpha for cost-complexity pruning (similar to REP)
    - min_impurity_decrease to control tree growth
    """

    def __init__(self, ccp_alpha=0.01, min_impurity_decrease=0.0):
        """
        Initialize REPTree classifier.

        Parameters:
        -----------
        ccp_alpha : float
            Complexity parameter for pruning
        min_impurity_decrease : float
            Minimum impurity decrease required for split
        """
        self.model = DecisionTreeClassifier(
            criterion='entropy',
            splitter='best',
            ccp_alpha=ccp_alpha,
            min_impurity_decrease=min_impurity_decrease,
            random_state=42
        )
        self.name = "REPTree"

    def fit(self, X, y):
        """Fit the model"""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

    def get_model(self):
        """Get the underlying sklearn model"""
        return self.model


class ModelEvaluator:
    """
    Model evaluator using 10-fold cross-validation as per the paper.
    """

    def __init__(self, n_folds=10):
        """
        Initialize evaluator.

        Parameters:
        -----------
        n_folds : int
            Number of folds for cross-validation (default: 10 as per paper)
        """
        self.n_folds = n_folds
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def evaluate_model(self, classifier, X, y):
        """
        Evaluate a classifier using 10-fold cross-validation.

        Parameters:
        -----------
        classifier : object
            Classifier object with fit and predict methods
        X : array-like
            Feature matrix
        y : array-like
            Target vector

        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted'
        }

        # Perform cross-validation
        cv_results = cross_validate(
            classifier.get_model(),
            X, y,
            cv=self.cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Calculate mean and std for each metric
        results = {
            'model_name': classifier.name,
            'accuracy_mean': cv_results['test_accuracy'].mean(),
            'accuracy_std': cv_results['test_accuracy'].std(),
            'precision_mean': cv_results['test_precision_weighted'].mean(),
            'precision_std': cv_results['test_precision_weighted'].std(),
            'recall_mean': cv_results['test_recall_weighted'].mean(),
            'recall_std': cv_results['test_recall_weighted'].std(),
            'f1_mean': cv_results['test_f1_weighted'].mean(),
            'f1_std': cv_results['test_f1_weighted'].std()
        }

        # Also calculate TP Rate and FP Rate manually for each fold
        tp_rates = []
        fp_rates = []

        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model
            classifier.get_model().fit(X_train, y_train)
            y_pred = classifier.get_model().predict(X_test)

            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Calculate rates per class and average
            n_classes = len(np.unique(y))

            if n_classes == 2:
                # Binary classification
                tn, fp, fn, tp = cm.ravel()
                tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                # Multi-class: calculate weighted average
                tp_rate = 0
                fp_rate = 0
                for i in range(n_classes):
                    tp = cm[i, i]
                    fn = cm[i, :].sum() - tp
                    fp = cm[:, i].sum() - tp
                    tn = cm.sum() - tp - fn - fp

                    class_tp_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
                    class_fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

                    # Weight by class frequency
                    weight = (tp + fn) / len(y_test)
                    tp_rate += class_tp_rate * weight
                    fp_rate += class_fp_rate * weight

            tp_rates.append(tp_rate)
            fp_rates.append(fp_rate)

        results['tp_rate_mean'] = np.mean(tp_rates)
        results['tp_rate_std'] = np.std(tp_rates)
        results['fp_rate_mean'] = np.mean(fp_rates)
        results['fp_rate_std'] = np.std(fp_rates)

        return results

    def compare_models(self, classifiers, X, y):
        """
        Compare multiple classifiers.

        Parameters:
        -----------
        classifiers : list
            List of classifier objects
        X : array-like
            Feature matrix
        y : array-like
            Target vector

        Returns:
        --------
        pd.DataFrame
            Comparison results
        """
        results = []

        for classifier in classifiers:
            print(f"\nEvaluating {classifier.name}...")
            result = self.evaluate_model(classifier, X, y)
            results.append(result)

        return pd.DataFrame(results)


def print_comparison_table(results_df):
    """
    Print comparison table similar to Tables IV and V in the paper.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe from compare_models
    """
    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE COMPARISON (10-Fold Cross-Validation)")
    print("=" * 100)
    print(f"\n{'Classifier':<15} {'TP Rate':<12} {'FP Rate':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 100)

    for _, row in results_df.iterrows():
        print(f"{row['model_name']:<15} "
              f"{row['tp_rate_mean']:.3f}        "
              f"{row['fp_rate_mean']:.3f}        "
              f"{row['precision_mean']:.3f}        "
              f"{row['recall_mean']:.3f}        "
              f"{row['f1_mean']:.3f}")

    print("-" * 100)

    # Find best model
    best_idx = results_df['accuracy_mean'].idxmax()
    best_model = results_df.loc[best_idx, 'model_name']

    print(f"\n{'Metric':<20} {' | '.join([f'{name:>12}' for name in results_df['model_name']])}")
    print("-" * 100)
    print(f"{'Accuracy':<20} {' | '.join([f'{acc:>12.4f}' for acc in results_df['accuracy_mean']])}")
    print(f"{'TP Rate':<20} {' | '.join([f'{tp:>12.4f}' for tp in results_df['tp_rate_mean']])}")
    print(f"{'FP Rate':<20} {' | '.join([f'{fp:>12.4f}' for fp in results_df['fp_rate_mean']])}")
    print(f"{'Precision':<20} {' | '.join([f'{p:>12.4f}' for p in results_df['precision_mean']])}")
    print(f"{'Recall':<20} {' | '.join([f'{r:>12.4f}' for r in results_df['recall_mean']])}")

    print("\n" + "=" * 100)
    print(f"BEST MODEL: {best_model}")
    print(f"  Accuracy: {results_df.loc[best_idx, 'accuracy_mean']:.4f}")
    print(f"  TP Rate: {results_df.loc[best_idx, 'tp_rate_mean']:.4f}")
    print(f"  FP Rate: {results_df.loc[best_idx, 'fp_rate_mean']:.4f}")
    print(f"  Precision: {results_df.loc[best_idx, 'precision_mean']:.4f}")
    print("=" * 100)


def main():
    """Main function to run all models"""
    from utils import get_data_dir, get_models_dir, get_results_dir

    print("=" * 100)
    print("DECISION TREE MODELS - IMPLEMENTATION AND EVALUATION")
    print("=" * 100)

    # Get directories
    data_dir = get_data_dir()
    models_dir = get_models_dir()
    results_dir = get_results_dir()

    # Load data (both filtered and unfiltered)
    print("\nLoading datasets...")
    df_full = pd.read_csv(data_dir / 'student_data_processed_numeric.csv')
    df_filtered = pd.read_csv(data_dir / 'student_data_filtered.csv')

    print(f"Full dataset: {df_full.shape}")
    print(f"Filtered dataset (top 40 attributes): {df_filtered.shape}")

    # Prepare data
    X_full = df_full.drop('Failed', axis=1).values
    y_full = df_full['Failed'].values

    X_filtered = df_filtered.drop('Failed', axis=1).values
    y_filtered = df_filtered['Failed'].values

    # Initialize classifiers
    classifiers = [
        J48Classifier(),
        RandomTreeClassifier(),
        REPTreeClassifier()
    ]

    # Initialize evaluator
    evaluator = ModelEvaluator(n_folds=10)

    # Evaluate on full dataset (without attribute filter)
    print("\n" + "=" * 100)
    print("EVALUATION WITHOUT ATTRIBUTE FILTER (All 60 attributes)")
    print("=" * 100)
    results_full = evaluator.compare_models(classifiers, X_full, y_full)
    print_comparison_table(results_full)

    # Re-initialize classifiers for filtered data
    classifiers_filtered = [
        J48Classifier(),
        RandomTreeClassifier(),
        REPTreeClassifier()
    ]

    # Evaluate on filtered dataset (with attribute filter - top 40)
    print("\n" + "=" * 100)
    print("EVALUATION WITH ATTRIBUTE FILTER (Top 40 attributes)")
    print("=" * 100)
    results_filtered = evaluator.compare_models(classifiers_filtered, X_filtered, y_filtered)
    print_comparison_table(results_filtered)

    # Save results
    print("\nSaving results...")
    results_full.to_csv(results_dir / 'model_comparison_full.csv', index=False)
    results_filtered.to_csv(results_dir / 'model_comparison_filtered.csv', index=False)

    # Train final models on full data for later use
    print("\nTraining final models on full filtered dataset...")
    for clf in classifiers_filtered:
        clf.fit(X_filtered, y_filtered)

        # Save model
        model_path = models_dir / f'{clf.name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f"  {clf.name} model saved to: {model_path}")

    print("\n" + "=" * 100)
    print("Model training and evaluation complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
