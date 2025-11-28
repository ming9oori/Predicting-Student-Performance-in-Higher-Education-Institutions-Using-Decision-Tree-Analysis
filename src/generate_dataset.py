"""
Generate sample dataset based on the paper's questionnaire structure.
This script creates a synthetic dataset with 161 responses (151 after cleaning)
matching the questionnaire described in Table I of the paper.
"""

import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_student_data(n_samples=161):
    """
    Generate student questionnaire responses based on the paper's structure.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 161 as per paper)

    Returns:
    --------
    pd.DataFrame
        DataFrame with 60 questions as columns
    """

    data = {}

    # Demographic Data
    data['Q1'] = np.random.choice(['IS', 'CS'], n_samples)
    data['Q2'] = np.random.choice(['18', '19', '20', '>20'], n_samples, p=[0.3, 0.3, 0.25, 0.15])
    data['Q3'] = np.random.choice(['1', '2', '3', '4'], n_samples, p=[0.25, 0.25, 0.25, 0.25])
    data['Q4'] = np.random.choice(['Female', 'Male'], n_samples)
    data['Q5'] = np.random.choice(['In Basra', 'Out of Basra'], n_samples, p=[0.7, 0.3])

    # Social Information
    data['Q6'] = np.random.choice(['Married', 'Single'], n_samples, p=[0.2, 0.8])
    data['Q7'] = np.random.choice(['YES', 'NO'], n_samples, p=[0.3, 0.7])
    data['Q8'] = np.random.choice(['YES', 'NO'], n_samples, p=[0.8, 0.2])
    data['Q9'] = np.random.choice(['YES', 'NO'], n_samples, p=[0.9, 0.1])
    data['Q10'] = np.random.choice(['YES', 'NO'], n_samples, p=[0.85, 0.15])
    data['Q11'] = np.random.choice(['YES', 'NO'], n_samples, p=[0.4, 0.6])

    # Academic Information
    data['Q12'] = np.random.choice(['0', '1', '2', '>2'], n_samples, p=[0.5, 0.25, 0.15, 0.1])
    data['Q13'] = np.random.choice(['0', '1-5', '5-10', '>10'], n_samples, p=[0.3, 0.4, 0.2, 0.1])
    data['Q14'] = np.random.choice(['<12', '13-17', '>17'], n_samples, p=[0.2, 0.5, 0.3])
    data['Q15'] = np.random.choice(['<60', '61-70', '71-80', '>80'], n_samples, p=[0.15, 0.3, 0.35, 0.2])
    data['Q16'] = np.random.choice(['<36', '36-71', '72-107', '>107'], n_samples, p=[0.25, 0.25, 0.25, 0.25])
    data['Q17'] = np.random.choice(['1', '2', '3', '>3'], n_samples, p=[0.25, 0.25, 0.25, 0.25])

    # Questions Q18-Q60 use Likert scale (1-5)
    # Study Skills (Q18-Q21)
    for q in range(18, 22):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.3, 0.3, 0.15])

    # Motivation (Q22-Q26)
    for q in range(22, 27):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.25, 0.35, 0.15])

    # Personal Relationship (Q27-Q30)
    for q in range(27, 31):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.3, 0.3, 0.15])

    # Health (Q31-Q34)
    for q in range(31, 35):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.2, 0.25, 0.25, 0.15])

    # Time Management (Q35-Q39)
    for q in range(35, 40):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.2, 0.3, 0.25, 0.1])

    # Money Management (Q40-Q44)
    for q in range(40, 45):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.2, 0.3, 0.25, 0.1])

    # Personal Purpose (Q45-Q49)
    for q in range(45, 50):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.25, 0.35, 0.15])

    # Career Planning (Q50-Q53)
    for q in range(50, 54):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.2, 0.3, 0.25, 0.1])

    # Resource Needs (Q54-Q56)
    for q in range(54, 57):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.2, 0.25, 0.3, 0.15, 0.1])

    # Self Esteem (Q57-Q60)
    for q in range(57, 61):
        data[f'Q{q}'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.15, 0.25, 0.35, 0.15])

    df = pd.DataFrame(data)

    # Introduce missing values (paper mentions 8 rows with empty values)
    # Randomly select 8 rows and introduce missing values
    missing_rows = np.random.choice(n_samples, 8, replace=False)
    missing_cols = np.random.choice(df.columns, len(missing_rows))

    for row, col in zip(missing_rows, missing_cols):
        df.loc[row, col] = np.nan

    return df


def main():
    """Generate and save the dataset"""
    import os
    from pathlib import Path

    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)

    print("Generating student performance dataset...")
    print("=" * 60)

    # Generate data
    df = generate_student_data(161)

    print(f"Total samples generated: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()

    # Save raw data
    output_path = data_dir / 'student_data_raw.csv'
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to: {output_path}")
    print()

    # Display sample
    print("Sample data (first 5 rows):")
    print(df.head())
    print()

    print("Data generation complete!")


if __name__ == "__main__":
    main()
