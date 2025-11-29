"""
Decision Tree Interpretation and Rule Extraction

This module provides in-depth analysis of decision trees:
- Extract and interpret decision rules
- Analyze decision paths
- Identify key decision points
- Visualize tree structure with annotations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import pickle
from utils import get_data_dir, get_models_dir, get_results_dir


class TreeInterpreter:
    """Interpret and analyze decision tree models"""

    def __init__(self, model_name='J48'):
        """
        Initialize tree interpreter

        Parameters:
        -----------
        model_name : str
            Name of the model to interpret (J48, RandomTree, or REPTree)
        """
        self.model_name = model_name
        self.results_dir = get_results_dir()
        self.models_dir = get_models_dir()
        self.data_dir = get_data_dir()

        # Load model
        with open(self.models_dir / f'{model_name}_model.pkl', 'rb') as f:
            self.clf = pickle.load(f)

        # Load data
        self.df_filtered = pd.read_csv(self.data_dir / 'student_data_filtered.csv')
        self.feature_names = [col for col in self.df_filtered.columns if col != 'Failed']
        self.class_names = ['Passed', 'Failed']

        # Get the tree
        self.tree_model = self.clf.get_model()

        print(f"Loaded {model_name} model")
        print(f"Tree depth: {self.tree_model.get_depth()}")
        print(f"Number of leaves: {self.tree_model.get_n_leaves()}")
        print(f"Number of features: {len(self.feature_names)}")

    def extract_rules(self, max_rules=20):
        """
        Extract decision rules from the tree

        Parameters:
        -----------
        max_rules : int
            Maximum number of rules to extract

        Returns:
        --------
        list
            List of decision rules
        """
        print("\n" + "="*80)
        print("EXTRACTING DECISION RULES")
        print("="*80)

        tree_model = self.tree_model.tree_
        feature_names = self.feature_names

        def recurse(node, depth, parent_rule):
            """Recursively extract rules from tree nodes"""
            indent = "  " * depth

            # Check if leaf node
            if tree_model.feature[node] == -2:  # Leaf node
                class_idx = np.argmax(tree_model.value[node])
                class_name = self.class_names[class_idx]
                samples = tree_model.n_node_samples[node]
                confidence = tree_model.value[node][0][class_idx] / samples

                rule = {
                    'depth': depth,
                    'path': parent_rule,
                    'prediction': class_name,
                    'samples': samples,
                    'confidence': confidence
                }
                return [rule]

            else:
                # Internal node
                feature = feature_names[tree_model.feature[node]]
                threshold = tree_model.threshold[node]

                # Left child (<=)
                left_rule = parent_rule + f" AND {feature} <= {threshold:.2f}"
                left_rules = recurse(tree_model.children_left[node], depth + 1, left_rule)

                # Right child (>)
                right_rule = parent_rule + f" AND {feature} > {threshold:.2f}"
                right_rules = recurse(tree_model.children_right[node], depth + 1, right_rule)

                return left_rules + right_rules

        # Extract all rules
        all_rules = recurse(0, 0, "IF TRUE")

        # Clean up rules
        for rule in all_rules:
            rule['path'] = rule['path'].replace("IF TRUE AND ", "IF ")

        # Sort by confidence and samples
        all_rules = sorted(all_rules, key=lambda x: (x['confidence'], x['samples']), reverse=True)

        # Save rules
        rules_df = pd.DataFrame(all_rules)
        rules_df.to_csv(self.results_dir / f'{self.model_name}_decision_rules.csv', index=False)

        # Print top rules
        print(f"\nExtracted {len(all_rules)} decision rules")
        print(f"\nTop {min(max_rules, len(all_rules))} most confident rules:\n")

        for idx, rule in enumerate(all_rules[:max_rules], 1):
            print(f"\n{idx}. {rule['path']}")
            print(f"   → THEN: {rule['prediction']}")
            print(f"   → Confidence: {rule['confidence']:.2%}")
            print(f"   → Samples: {rule['samples']}")

        return all_rules

    def analyze_feature_paths(self):
        """Analyze which features appear most often in decision paths"""
        print("\n" + "="*80)
        print("FEATURE PATH ANALYSIS")
        print("="*80)

        tree_model = self.tree_model.tree_

        feature_usage = {name: 0 for name in self.feature_names}

        def count_features(node):
            """Count feature usage in tree"""
            if tree_model.feature[node] != -2:  # Not a leaf
                feature = self.feature_names[tree_model.feature[node]]
                feature_usage[feature] += 1

                # Recurse
                count_features(tree_model.children_left[node])
                count_features(tree_model.children_right[node])

        count_features(0)

        # Convert to dataframe
        usage_df = pd.DataFrame(list(feature_usage.items()),
                               columns=['Feature', 'Usage_Count'])
        usage_df = usage_df[usage_df['Usage_Count'] > 0].sort_values('Usage_Count', ascending=False)

        print(f"\nFeatures used in decision tree: {len(usage_df)}")
        print(f"\nTop 15 most frequently used features:\n")
        print(usage_df.head(15).to_string(index=False))

        # Visualize
        fig, ax = plt.subplots(figsize=(12, 8))

        top_features = usage_df.head(20)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

        ax.barh(range(len(top_features)), top_features['Usage_Count'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Number of Times Used in Tree', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Usage Frequency in {self.model_name} Decision Tree',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['Usage_Count'], i, f" {int(row['Usage_Count'])}",
                   va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{self.model_name}_feature_usage.png',
                   dpi=300, bbox_inches='tight')

        usage_df.to_csv(self.results_dir / f'{self.model_name}_feature_usage.csv', index=False)
        print(f"\n✓ Feature usage analysis saved")

        return usage_df

    def visualize_tree_detailed(self, max_depth=4):
        """Create detailed tree visualization with annotations"""
        print("\n" + "="*80)
        print(f"VISUALIZING TREE (max_depth={max_depth})")
        print("="*80)

        fig, ax = plt.subplots(figsize=(30, 20))

        tree.plot_tree(self.tree_model,
                      feature_names=self.feature_names,
                      class_names=self.class_names,
                      filled=True,
                      rounded=True,
                      fontsize=11,
                      max_depth=max_depth,
                      proportion=True,
                      precision=2,
                      ax=ax)

        ax.set_title(f'{self.model_name} Decision Tree - Detailed View (Max Depth = {max_depth})',
                    fontsize=18, fontweight='bold', pad=30)

        plt.tight_layout()
        plt.savefig(self.results_dir / f'{self.model_name}_tree_detailed.png',
                   dpi=300, bbox_inches='tight')

        print(f"✓ Detailed tree visualization saved")

        return fig

    def analyze_decision_paths(self, n_samples=10):
        """Analyze specific decision paths for sample students"""
        print("\n" + "="*80)
        print(f"ANALYZING DECISION PATHS FOR {n_samples} SAMPLE STUDENTS")
        print("="*80)

        # Get sample data
        X = self.df_filtered.drop('Failed', axis=1).values[:n_samples]
        y_true = self.df_filtered['Failed'].values[:n_samples]
        y_pred = self.tree_model.predict(X)

        # Get decision paths
        decision_paths = self.tree_model.decision_path(X)

        path_analysis = []

        for i in range(n_samples):
            path = decision_paths[i].toarray()[0]
            nodes_in_path = np.where(path == 1)[0]

            print(f"\n{'='*70}")
            print(f"Student {i+1}:")
            print(f"  True class: {self.class_names[y_true[i]]}")
            print(f"  Predicted: {self.class_names[y_pred[i]]}")
            print(f"  Correct: {'✓' if y_true[i] == y_pred[i] else '✗'}")
            print(f"\n  Decision path ({len(nodes_in_path)} nodes):")

            path_str = []

            for node_id in nodes_in_path:
                if self.tree_model.tree_.feature[node_id] != -2:  # Not a leaf
                    feature = self.feature_names[self.tree_model.tree_.feature[node_id]]
                    threshold = self.tree_model.tree_.threshold[node_id]
                    value = X[i][self.tree_model.tree_.feature[node_id]]

                    if value <= threshold:
                        decision = f"{feature} = {value:.2f} <= {threshold:.2f}"
                    else:
                        decision = f"{feature} = {value:.2f} > {threshold:.2f}"

                    print(f"    → {decision}")
                    path_str.append(decision)

            path_analysis.append({
                'Student': i+1,
                'True_Class': self.class_names[y_true[i]],
                'Predicted_Class': self.class_names[y_pred[i]],
                'Correct': y_true[i] == y_pred[i],
                'Path_Length': len(nodes_in_path),
                'Decision_Path': ' → '.join(path_str)
            })

        # Save analysis
        pd.DataFrame(path_analysis).to_csv(
            self.results_dir / f'{self.model_name}_sample_paths.csv', index=False)

        print(f"\n✓ Decision path analysis saved")

        return path_analysis

    def generate_insights(self):
        """Generate actionable insights from tree analysis"""
        print("\n" + "="*80)
        print("GENERATING ACTIONABLE INSIGHTS")
        print("="*80)

        insights = []

        # Feature importance
        importance = self.tree_model.feature_importances_
        important_features = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)

        insights.append("# Decision Tree Insights and Recommendations\n")
        insights.append("="*80 + "\n\n")

        insights.append("## 1. Most Important Decision Factors\n\n")
        insights.append("The following factors have the highest impact on student success:\n\n")

        for idx, row in important_features.head(10).iterrows():
            if row['Importance'] > 0:
                insights.append(f"{idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")

        insights.append("\n## 2. Key Decision Points\n\n")
        insights.append("Critical thresholds identified by the decision tree:\n\n")

        # Load feature usage
        usage_df = pd.read_csv(self.results_dir / f'{self.model_name}_feature_usage.csv')

        insights.append("Most frequently used features in decision making:\n")
        for idx, row in usage_df.head(5).iterrows():
            insights.append(f"- {row['Feature']}: Used {int(row['Usage_Count'])} times\n")

        insights.append("\n## 3. Recommendations for Students\n\n")
        insights.append("Based on the decision tree analysis:\n\n")
        insights.append("### For At-Risk Students:\n")
        insights.append("- Focus on improving top 5 critical factors\n")
        insights.append("- Seek help early if struggling with key areas\n")
        insights.append("- Regular monitoring of progress\n\n")

        insights.append("### For All Students:\n")
        insights.append("- Maintain consistent study habits\n")
        insights.append("- Develop strong time management skills\n")
        insights.append("- Stay motivated and engaged\n")

        insights.append("\n## 4. Recommendations for Educators\n\n")
        insights.append("- Design interventions targeting high-impact factors\n")
        insights.append("- Early identification using decision tree predictions\n")
        insights.append("- Personalized support based on decision paths\n")
        insights.append("- Monitor key indicators throughout semester\n")

        # Save insights
        with open(self.results_dir / f'{self.model_name}_insights.txt', 'w') as f:
            f.writelines(insights)

        print("\n".join(insights))
        print(f"\n✓ Insights saved to {self.model_name}_insights.txt")


def main():
    """Run comprehensive tree interpretation"""
    print("="*80)
    print("DECISION TREE INTERPRETATION AND ANALYSIS")
    print("="*80)

    # Interpret J48 model (best performing)
    interpreter = TreeInterpreter('J48')

    # Extract and analyze rules
    rules = interpreter.extract_rules(max_rules=20)

    # Analyze feature paths
    usage = interpreter.analyze_feature_paths()

    # Visualize tree
    interpreter.visualize_tree_detailed(max_depth=5)

    # Analyze sample paths
    paths = interpreter.analyze_decision_paths(n_samples=10)

    # Generate insights
    interpreter.generate_insights()

    print("\n" + "="*80)
    print("✓ TREE INTERPRETATION COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  - {interpreter.model_name}_decision_rules.csv")
    print(f"  - {interpreter.model_name}_feature_usage.csv")
    print(f"  - {interpreter.model_name}_feature_usage.png")
    print(f"  - {interpreter.model_name}_tree_detailed.png")
    print(f"  - {interpreter.model_name}_sample_paths.csv")
    print(f"  - {interpreter.model_name}_insights.txt")


if __name__ == "__main__":
    main()
