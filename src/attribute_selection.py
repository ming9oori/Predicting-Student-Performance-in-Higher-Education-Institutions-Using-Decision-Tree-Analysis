"""
Attribute selection module using correlation analysis.

This module implements the CorrelationAttributeEval approach from the paper:
- Evaluates correlation (Pearson's) between each attribute and the class
- Ranks attributes by correlation strength
- Identifies low-correlation attributes for removal
- Paper states: "Last twenty questions will be removed to increase accuracy"
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


class AttributeSelector:
    """
    Attribute selector using correlation analysis following the paper's methodology.
    """

    def __init__(self, df_numeric, target_column='Failed'):
        """
        Initialize attribute selector.

        Parameters:
        -----------
        df_numeric : pd.DataFrame
            DataFrame with numeric encoding
        target_column : str
            Name of the target variable column
        """
        self.df = df_numeric
        self.target_column = target_column
        self.correlations = None
        self.selected_features = None

    def calculate_correlations(self):
        """
        Calculate Pearson correlation between each attribute and target variable.

        Returns:
        --------
        pd.DataFrame
            DataFrame with attributes and their correlation values
        """
        print("Calculating Pearson correlations with target variable...")

        # Get all feature columns (exclude target)
        feature_columns = [col for col in self.df.columns if col != self.target_column]

        correlations = []

        for col in feature_columns:
            # Calculate Pearson correlation
            corr, p_value = pearsonr(self.df[col], self.df[self.target_column])

            correlations.append({
                'Attribute': col,
                'Correlation': abs(corr),  # Use absolute value for ranking
                'Correlation_Signed': corr,  # Keep signed value for analysis
                'P_Value': p_value
            })

        # Create DataFrame and sort by correlation (descending)
        self.correlations = pd.DataFrame(correlations)
        self.correlations = self.correlations.sort_values('Correlation', ascending=False)
        self.correlations['Rank'] = range(1, len(self.correlations) + 1)

        print(f"Correlations calculated for {len(feature_columns)} attributes")

        return self.correlations

    def print_correlation_table(self, top_n=None):
        """
        Print correlation table similar to Table III in the paper.

        Parameters:
        -----------
        top_n : int, optional
            Number of top correlations to display
        """
        print("\n" + "=" * 80)
        print("ATTRIBUTE CORRELATION ANALYSIS (Similar to Table III in paper)")
        print("=" * 80)

        if top_n is None:
            display_df = self.correlations
        else:
            display_df = self.correlations.head(top_n)

        print(f"\n{'Rank':<6} {'Attribute':<12} {'Correlation':<15} {'P-Value':<12}")
        print("-" * 80)

        for _, row in display_df.iterrows():
            print(f"{int(row['Rank']):<6} {row['Attribute']:<12} "
                  f"{row['Correlation']:.4f} ({row['Correlation_Signed']:+.4f})  {row['P_Value']:.4f}")

        print("-" * 80)

    def select_top_attributes(self, n_features=40):
        """
        Select top N attributes based on correlation.

        According to the paper:
        "The last twenty questions will be removed to increase the accuracy of the result."
        This means keeping the top 40 questions (60 - 20 = 40).

        Parameters:
        -----------
        n_features : int
            Number of top features to select (default: 40)

        Returns:
        --------
        list
            List of selected attribute names
        """
        print(f"\nSelecting top {n_features} attributes...")

        self.selected_features = self.correlations.head(n_features)['Attribute'].tolist()

        print(f"Selected {len(self.selected_features)} attributes")
        print(f"Removed {len(self.correlations) - len(self.selected_features)} low-correlation attributes")

        return self.selected_features

    def get_removed_attributes(self, n_features=40):
        """
        Get list of removed (low-correlation) attributes.

        Parameters:
        -----------
        n_features : int
            Number of top features to keep

        Returns:
        --------
        list
            List of removed attribute names
        """
        removed = self.correlations.tail(len(self.correlations) - n_features)['Attribute'].tolist()
        return removed

    def visualize_correlations(self, output_path=None):
        """
        Create visualization of attribute correlations.

        Parameters:
        -----------
        output_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Bar plot of all correlations
        ax1 = axes[0]
        colors = ['green' if i < 40 else 'red' for i in range(len(self.correlations))]
        ax1.bar(range(len(self.correlations)), self.correlations['Correlation'], color=colors, alpha=0.7)
        ax1.axhline(y=0.1, color='blue', linestyle='--', label='Threshold (example)')
        ax1.set_xlabel('Attribute Rank', fontsize=12)
        ax1.set_ylabel('Absolute Correlation', fontsize=12)
        ax1.set_title('Attribute Correlations with Target Variable (Sorted)', fontsize=14, fontweight='bold')
        ax1.legend(['Top 40 (Selected)', 'Bottom 20 (Removed)', 'Threshold'], loc='upper right')
        ax1.grid(axis='y', alpha=0.3)

        # Plot 2: Top 20 and Bottom 20 comparison
        ax2 = axes[1]
        top_20 = self.correlations.head(20)
        bottom_20 = self.correlations.tail(20)

        x1 = np.arange(len(top_20))
        x2 = np.arange(len(bottom_20))

        width = 0.4

        ax2.barh(x1, top_20['Correlation'], height=width, label='Top 20', color='green', alpha=0.7)
        ax2.barh(x2 + width, bottom_20['Correlation'], height=width, label='Bottom 20', color='red', alpha=0.7)

        ax2.set_yticks(x1 + width / 2)
        ax2.set_yticklabels([f"Rank {i+1}" for i in range(20)])
        ax2.set_xlabel('Absolute Correlation', fontsize=12)
        ax2.set_ylabel('Rank', fontsize=12)
        ax2.set_title('Top 20 vs Bottom 20 Attributes by Correlation', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_path}")

        return fig

    def create_filtered_dataset(self, n_features=40):
        """
        Create new dataset with only selected features.

        Parameters:
        -----------
        n_features : int
            Number of top features to keep

        Returns:
        --------
        pd.DataFrame
            Filtered dataset
        """
        if self.selected_features is None:
            self.select_top_attributes(n_features)

        filtered_df = self.df[self.selected_features + [self.target_column]].copy()

        print(f"\nFiltered dataset shape: {filtered_df.shape}")
        print(f"Original dataset shape: {self.df.shape}")

        return filtered_df


def main():
    """Main attribute selection function"""
    from utils import get_data_dir, get_results_dir

    print("=" * 80)
    print("ATTRIBUTE SELECTION USING CORRELATION ANALYSIS")
    print("=" * 80)

    # Get directories
    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # Load numeric processed data
    df_numeric = pd.read_csv(data_dir / 'student_data_processed_numeric.csv')
    print(f"Data loaded: {df_numeric.shape}")
    print()

    # Initialize attribute selector
    selector = AttributeSelector(df_numeric, target_column='Failed')

    # Calculate correlations
    correlations_df = selector.calculate_correlations()

    # Print correlation table (similar to Table III)
    selector.print_correlation_table()

    # Print statistics
    print("\n" + "=" * 80)
    print("CORRELATION STATISTICS")
    print("=" * 80)
    print(f"Mean correlation: {correlations_df['Correlation'].mean():.4f}")
    print(f"Median correlation: {correlations_df['Correlation'].median():.4f}")
    print(f"Max correlation: {correlations_df['Correlation'].max():.4f}")
    print(f"Min correlation: {correlations_df['Correlation'].min():.4f}")
    print()

    # Select top 40 attributes (removing bottom 20 as per paper)
    selected_features = selector.select_top_attributes(n_features=40)

    print("\nTop 10 selected attributes:")
    for i, attr in enumerate(selected_features[:10], 1):
        corr = correlations_df[correlations_df['Attribute'] == attr]['Correlation'].values[0]
        print(f"  {i}. {attr}: {corr:.4f}")

    print("\nBottom 10 removed attributes:")
    removed = selector.get_removed_attributes(n_features=40)
    for i, attr in enumerate(removed[-10:], 1):
        corr = correlations_df[correlations_df['Attribute'] == attr]['Correlation'].values[0]
        print(f"  {i}. {attr}: {corr:.4f}")

    # Create filtered dataset
    filtered_df = selector.create_filtered_dataset(n_features=40)

    # Save results
    print("\nSaving results...")
    correlations_df.to_csv(results_dir / 'attribute_correlations.csv', index=False)
    filtered_df.to_csv(data_dir / 'student_data_filtered.csv', index=False)

    # Create visualizations
    print("\nCreating visualizations...")
    selector.visualize_correlations(str(results_dir / 'correlation_analysis.png'))

    print("\n" + "=" * 80)
    print("Attribute selection complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
