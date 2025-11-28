"""
Reliability analysis module using Cronbach's Alpha.

According to the paper (Table II):
- Cronbach's alpha: 0.85
- Number of items: 60
- Number of respondents: 161 (100%)

This module calculates Cronbach's alpha to measure internal consistency.
"""

import pandas as pd
import numpy as np


def calculate_cronbachs_alpha(df, columns=None):
    """
    Calculate Cronbach's Alpha for internal consistency reliability.

    Cronbach's Alpha formula:
    α = (K / (K-1)) * (1 - (Σσ²ᵢ / σ²ₜ))

    where:
    - K is the number of items
    - σ²ᵢ is the variance of item i
    - σ²ₜ is the variance of the total score

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the questionnaire data
    columns : list, optional
        List of columns to include. If None, uses all numeric columns.

    Returns:
    --------
    float
        Cronbach's alpha value
    dict
        Additional statistics
    """
    if columns is None:
        # Use all numeric columns except the target variable
        columns = [col for col in df.columns if col not in ['Failed']]

    # Select only the specified columns
    data = df[columns]

    # Number of items
    K = len(columns)

    # Calculate variance of each item
    item_variances = data.var(axis=0, ddof=1)
    sum_item_variances = item_variances.sum()

    # Calculate total score for each respondent
    total_scores = data.sum(axis=1)

    # Calculate variance of total scores
    total_variance = total_scores.var(ddof=1)

    # Calculate Cronbach's Alpha
    if total_variance == 0:
        alpha = 0
    else:
        alpha = (K / (K - 1)) * (1 - (sum_item_variances / total_variance))

    stats = {
        'alpha': alpha,
        'n_items': K,
        'n_respondents': len(df),
        'mean_item_variance': item_variances.mean(),
        'total_variance': total_variance,
        'sum_item_variances': sum_item_variances
    }

    return alpha, stats


def calculate_alpha_if_item_deleted(df, columns=None):
    """
    Calculate Cronbach's Alpha if each item is deleted.
    This helps identify items that reduce reliability.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the questionnaire data
    columns : list, optional
        List of columns to include

    Returns:
    --------
    pd.DataFrame
        DataFrame with alpha values if each item is deleted
    """
    if columns is None:
        columns = [col for col in df.columns if col not in ['Failed']]

    results = []

    for col in columns:
        # Calculate alpha without this column
        remaining_cols = [c for c in columns if c != col]
        alpha, _ = calculate_cronbachs_alpha(df, remaining_cols)

        results.append({
            'Item': col,
            'Alpha_if_deleted': alpha
        })

    return pd.DataFrame(results).sort_values('Alpha_if_deleted', ascending=False)


def reliability_analysis_report(df):
    """
    Generate comprehensive reliability analysis report.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the questionnaire data

    Returns:
    --------
    dict
        Dictionary containing all reliability metrics
    """
    print("=" * 60)
    print("RELIABILITY ANALYSIS (Cronbach's Alpha)")
    print("=" * 60)

    # Calculate overall Cronbach's Alpha
    columns = [col for col in df.columns if col not in ['Failed']]
    alpha, stats = calculate_cronbachs_alpha(df, columns)

    print(f"\nOverall Cronbach's Alpha: {alpha:.3f}")
    print(f"Number of items: {stats['n_items']}")
    print(f"Number of respondents: {stats['n_respondents']}")
    print()

    # Interpretation
    print("Interpretation:")
    if alpha >= 0.9:
        interpretation = "Excellent"
    elif alpha >= 0.8:
        interpretation = "Good"
    elif alpha >= 0.7:
        interpretation = "Acceptable"
    elif alpha >= 0.6:
        interpretation = "Questionable"
    elif alpha >= 0.5:
        interpretation = "Poor"
    else:
        interpretation = "Unacceptable"

    print(f"  α = {alpha:.3f} → {interpretation} internal consistency")
    print(f"  (Paper reported: α = 0.85 → Good)")
    print()

    # Additional statistics
    print("Additional Statistics:")
    print(f"  Mean item variance: {stats['mean_item_variance']:.4f}")
    print(f"  Total variance: {stats['total_variance']:.4f}")
    print(f"  Sum of item variances: {stats['sum_item_variances']:.4f}")
    print()

    # Calculate alpha if item deleted (top 10 items that improve reliability)
    print("Top 10 items that would improve reliability if deleted:")
    alpha_if_deleted = calculate_alpha_if_item_deleted(df, columns)
    print(alpha_if_deleted.head(10).to_string(index=False))
    print()

    # Save detailed results
    results = {
        'overall_alpha': alpha,
        'stats': stats,
        'alpha_if_deleted': alpha_if_deleted,
        'interpretation': interpretation
    }

    return results


def main():
    """Main reliability analysis function"""
    from utils import get_data_dir, get_results_dir

    print("Loading processed data for reliability analysis...")

    # Get directories
    data_dir = get_data_dir()
    results_dir = get_results_dir()

    # Load numeric version of processed data
    df = pd.read_csv(data_dir / 'student_data_processed_numeric.csv')

    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print()

    # Perform reliability analysis
    results = reliability_analysis_report(df)

    # Save results
    print("Saving reliability analysis results...")
    results['alpha_if_deleted'].to_csv(results_dir / 'reliability_alpha_if_deleted.csv', index=False)

    # Save summary statistics
    summary = pd.DataFrame([{
        'Cronbachs_Alpha': results['overall_alpha'],
        'N_Items': results['stats']['n_items'],
        'N_Respondents': results['stats']['n_respondents'],
        'Interpretation': results['interpretation'],
        'Paper_Reported_Alpha': 0.85
    }])
    summary.to_csv(results_dir / 'reliability_summary.csv', index=False)

    print("\nReliability analysis complete!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
