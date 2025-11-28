"""
Result visualization module.

This module creates visualizations similar to those presented in the paper:
- Performance comparison charts (Figure 2)
- Decision tree visualization (Figure 3)
- Additional analysis charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import pickle


def load_models():
    """Load trained models from pickle files"""
    from utils import get_models_dir

    models = {}
    model_names = ['J48', 'RandomTree', 'REPTree']
    models_dir = get_models_dir()

    for name in model_names:
        try:
            with open(models_dir / f'{name}_model.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {name} model not found")

    return models


def plot_performance_comparison(results_full, results_filtered, output_path=None):
    """
    Create performance comparison chart similar to Figure 2 in the paper.

    Parameters:
    -----------
    results_full : pd.DataFrame
        Results without attribute filter
    results_filtered : pd.DataFrame
        Results with attribute filter
    output_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['tp_rate_mean', 'fp_rate_mean', 'precision_mean', 'recall_mean']
    metric_names = ['TP Rate', 'FP Rate', 'Precision', 'Recall']

    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]

        x = np.arange(len(results_full))
        width = 0.35

        # Plot without filter
        bars1 = ax.bar(x - width/2, results_full[metric], width,
                      label='Without Attribute Filter', alpha=0.8, color='steelblue')

        # Plot with filter
        bars2 = ax.bar(x + width/2, results_filtered[metric], width,
                      label='With Attribute Filter', alpha=0.8, color='coral')

        ax.set_xlabel('Classifier', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_full['model_name'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to: {output_path}")

    return fig


def plot_detailed_comparison(results_full, results_filtered, output_path=None):
    """
    Create detailed comparison chart showing all metrics together.

    Parameters:
    -----------
    results_full : pd.DataFrame
        Results without attribute filter
    results_filtered : pd.DataFrame
        Results with attribute filter
    output_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics = ['tp_rate_mean', 'fp_rate_mean', 'precision_mean', 'recall_mean']
    metric_labels = ['TP Rate', 'FP Rate', 'Precision', 'Recall']

    x = np.arange(len(metrics))
    width = 0.12

    models = results_filtered['model_name'].tolist()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, model in enumerate(models):
        # Get values for this model
        values_full = results_full[results_full['model_name'] == model][metrics].values[0]
        values_filtered = results_filtered[results_filtered['model_name'] == model][metrics].values[0]

        offset_full = (i - len(models)/2) * width * 2
        offset_filtered = offset_full + width

        ax.bar(x + offset_full, values_full, width, label=f'{model} (Full)',
               alpha=0.6, color=colors[i])
        ax.bar(x + offset_filtered, values_filtered, width, label=f'{model} (Filtered)',
               alpha=1.0, color=colors[i])

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Model Performance Comparison\n(Full vs Filtered Attributes)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Detailed comparison saved to: {output_path}")

    return fig


def plot_decision_tree(model_name='J48', max_depth=5, output_path=None):
    """
    Visualize decision tree similar to Figure 3 in the paper.

    Parameters:
    -----------
    model_name : str
        Name of the model to visualize
    max_depth : int
        Maximum depth to display
    output_path : str, optional
        Path to save the figure
    """
    from utils import get_models_dir, get_data_dir

    models_dir = get_models_dir()
    data_dir = get_data_dir()

    # Load model
    try:
        with open(models_dir / f'{model_name}_model.pkl', 'rb') as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_name} not found")
        return None

    # Load feature names
    df = pd.read_csv(data_dir / 'student_data_filtered.csv')
    feature_names = [col for col in df.columns if col != 'Failed']
    class_names = ['Passed', 'Failed']

    # Create figure
    fig, ax = plt.subplots(figsize=(25, 15))

    # Plot tree
    tree.plot_tree(clf.get_model(),
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  max_depth=max_depth,
                  ax=ax)

    ax.set_title(f'{model_name} Decision Tree (max_depth={max_depth})',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Decision tree visualization saved to: {output_path}")

    return fig


def plot_feature_importance(model_name='J48', top_n=20, output_path=None):
    """
    Plot feature importance for the selected model.

    Parameters:
    -----------
    model_name : str
        Name of the model
    top_n : int
        Number of top features to display
    output_path : str, optional
        Path to save the figure
    """
    from utils import get_models_dir, get_data_dir

    models_dir = get_models_dir()
    data_dir = get_data_dir()

    # Load model
    try:
        with open(models_dir / f'{model_name}_model.pkl', 'rb') as f:
            clf = pickle.load(f)
    except FileNotFoundError:
        print(f"Model {model_name} not found")
        return None

    # Load feature names
    df = pd.read_csv(data_dir / 'student_data_filtered.csv')
    feature_names = [col for col in df.columns if col != 'Failed']

    # Get feature importance
    importance = clf.get_model().feature_importances_

    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    # Plot top N features
    fig, ax = plt.subplots(figsize=(12, 8))

    top_features = importance_df.head(top_n)

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

    ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances ({model_name})',
                fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['Importance'], i, f" {row['Importance']:.4f}",
               va='center', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance saved to: {output_path}")

    # Save to CSV
    csv_path = output_path.replace('.png', '.csv') if output_path else None
    if csv_path:
        importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance data saved to: {csv_path}")

    return fig


def create_summary_report():
    """
    Create a comprehensive summary report comparing with paper results.
    """
    from utils import get_results_dir

    results_dir = get_results_dir()

    print("\n" + "=" * 100)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 100)

    # Load results
    results_full = pd.read_csv(results_dir / 'model_comparison_full.csv')
    results_filtered = pd.read_csv(results_dir / 'model_comparison_filtered.csv')

    print("\n1. PERFORMANCE WITHOUT ATTRIBUTE FILTER")
    print("-" * 100)
    print(results_full.to_string(index=False))

    print("\n2. PERFORMANCE WITH ATTRIBUTE FILTER (Top 40 attributes)")
    print("-" * 100)
    print(results_filtered.to_string(index=False))

    print("\n3. COMPARISON WITH PAPER RESULTS (Table IV)")
    print("-" * 100)
    print("Paper Results (Without Attribute Filter):")
    print("  J48:        TP=0.529, FP=0.485, Precision=0.539, Recall=0.529")
    print("  RandomTree: TP=0.608, FP=0.442, Precision=0.601, Recall=0.608")
    print("  RepTree:    TP=0.621, FP=0.448, Precision=0.609, Recall=0.621")
    print()
    print("Our Results (Without Attribute Filter):")
    for _, row in results_full.iterrows():
        print(f"  {row['model_name']:<11}: TP={row['tp_rate_mean']:.3f}, "
              f"FP={row['fp_rate_mean']:.3f}, "
              f"Precision={row['precision_mean']:.3f}, "
              f"Recall={row['recall_mean']:.3f}")

    print("\n4. COMPARISON WITH PAPER RESULTS (Table V - After Attribute Removal)")
    print("-" * 100)
    print("Paper Results (With Attribute Filter):")
    print("  J48:        TP=0.634, FP=0.409, Precision=0.629, Recall=0.634")
    print("  RandomTree: TP=0.614, FP=0.423, Precision=0.597, Recall=0.614")
    print("  RepTree:    TP=0.601, FP=0.488, Precision=0.583, Recall=0.601")
    print()
    print("Our Results (With Attribute Filter):")
    for _, row in results_filtered.iterrows():
        print(f"  {row['model_name']:<11}: TP={row['tp_rate_mean']:.3f}, "
              f"FP={row['fp_rate_mean']:.3f}, "
              f"Precision={row['precision_mean']:.3f}, "
              f"Recall={row['recall_mean']:.3f}")

    print("\n5. BEST MODEL SELECTION")
    print("-" * 100)
    best_model_paper = "J48"
    best_idx = results_filtered['accuracy_mean'].idxmax()
    best_model_ours = results_filtered.loc[best_idx, 'model_name']

    print(f"Paper's best model: {best_model_paper}")
    print(f"Our best model:     {best_model_ours}")
    print()
    print("Paper's conclusion:")
    print("  'J48 algorithm was considered as the best algorithm based on its performance'")
    print("  'compared with the Random Tree and RepTree algorithms.'")

    print("\n" + "=" * 100)


def main():
    """Main visualization function"""
    from utils import get_results_dir

    results_dir = get_results_dir()

    print("=" * 100)
    print("RESULT VISUALIZATION AND ANALYSIS")
    print("=" * 100)

    # Load results
    results_full = pd.read_csv(results_dir / 'model_comparison_full.csv')
    results_filtered = pd.read_csv(results_dir / 'model_comparison_filtered.csv')

    print("\nCreating visualizations...")

    # 1. Performance comparison (similar to Figure 2)
    print("\n1. Creating performance comparison chart...")
    plot_performance_comparison(results_full, results_filtered,
                               str(results_dir / 'figure2_performance_comparison.png'))

    # 2. Detailed comparison
    print("2. Creating detailed comparison chart...")
    plot_detailed_comparison(results_full, results_filtered,
                            str(results_dir / 'detailed_comparison.png'))

    # 3. Decision tree visualization (similar to Figure 3)
    print("3. Creating decision tree visualization (J48)...")
    plot_decision_tree('J48', max_depth=5,
                      output_path=str(results_dir / 'figure3_j48_tree.png'))

    # 4. Feature importance
    print("4. Creating feature importance chart (J48)...")
    plot_feature_importance('J48', top_n=20,
                           output_path=str(results_dir / 'j48_feature_importance.png'))

    # 5. Summary report
    print("\n5. Generating summary report...")
    create_summary_report()

    print("\n" + "=" * 100)
    print("All visualizations created successfully!")
    print(f"Results saved to: {results_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
