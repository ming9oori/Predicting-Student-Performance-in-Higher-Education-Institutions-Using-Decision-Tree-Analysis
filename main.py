#!/usr/bin/env python3
"""
Main execution script for replicating the paper:
"Predicting Student Performance in Higher Education Institutions Using Decision Tree Analysis"
by Alaa Khalaf Hamoud, Ali Salah Hashim, Wid Aqeel Awadh

This script runs the complete pipeline:
1. Generate sample dataset
2. Data preprocessing
3. Reliability analysis (Cronbach's alpha)
4. Attribute selection (correlation analysis)
5. Model training and evaluation
6. Result visualization

Usage:
    python main.py [--skip-data-generation]

Options:
    --skip-data-generation    Skip data generation step (use existing data)
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import modules
from generate_dataset import main as generate_data
from preprocess_data import main as preprocess_data
from reliability_analysis import main as reliability_analysis
from attribute_selection import main as attribute_selection
from decision_tree_models import main as train_models
from visualize_results import main as visualize_results


def print_header(text):
    """Print formatted section header"""
    print("\n" + "=" * 100)
    print(text.center(100))
    print("=" * 100 + "\n")


def ensure_directories():
    """Ensure all required directories exist"""
    directories = ['data', 'models', 'results', 'notebooks', 'src']

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

    print("✓ All required directories created/verified")


def run_pipeline(skip_data_generation=False):
    """
    Run the complete experimental pipeline.

    Parameters:
    -----------
    skip_data_generation : bool
        If True, skip data generation and use existing data
    """
    print_header("REPLICATING PAPER EXPERIMENT")
    print("Paper: Predicting Student Performance in Higher Education Institutions")
    print("       Using Decision Tree Analysis")
    print("Authors: Alaa Khalaf Hamoud, Ali Salah Hashim, Wid Aqeel Awadh")
    print("Published: February 2018")
    print()

    # Ensure directories exist
    ensure_directories()
    print()

    try:
        # Step 1: Generate dataset
        if not skip_data_generation:
            print_header("STEP 1: GENERATING SAMPLE DATASET")
            print("Generating 161 student responses to 60-question survey...")
            generate_data()
            print("\n✓ Data generation complete")
        else:
            print_header("STEP 1: SKIPPED (Using existing data)")

        # Step 2: Data preprocessing
        print_header("STEP 2: DATA PREPROCESSING")
        print("Removing empty values and creating 'Failed' column...")
        preprocess_data()
        print("\n✓ Data preprocessing complete")

        # Step 3: Reliability analysis
        print_header("STEP 3: RELIABILITY ANALYSIS")
        print("Calculating Cronbach's Alpha for internal consistency...")
        reliability_analysis()
        print("\n✓ Reliability analysis complete")

        # Step 4: Attribute selection
        print_header("STEP 4: ATTRIBUTE SELECTION")
        print("Analyzing correlations and selecting top 40 attributes...")
        attribute_selection()
        print("\n✓ Attribute selection complete")

        # Step 5: Model training and evaluation
        print_header("STEP 5: MODEL TRAINING AND EVALUATION")
        print("Training J48, Random Tree, and REPTree classifiers...")
        print("Using 10-fold cross-validation...")
        train_models()
        print("\n✓ Model training and evaluation complete")

        # Step 6: Visualization
        print_header("STEP 6: RESULT VISUALIZATION")
        print("Creating charts and decision tree visualizations...")
        visualize_results()
        print("\n✓ Visualization complete")

        # Final summary
        print_header("EXPERIMENT COMPLETE!")
        print("All results have been saved to the following directories:")
        print("  - data/       : Generated and processed datasets")
        print("  - models/     : Trained decision tree models")
        print("  - results/    : Analysis results, charts, and visualizations")
        print()
        print("Key files:")
        print("  - results/model_comparison_full.csv       : Full dataset results")
        print("  - results/model_comparison_filtered.csv   : Filtered dataset results")
        print("  - results/figure2_performance_comparison.png : Performance charts")
        print("  - results/figure3_j48_tree.png           : J48 decision tree")
        print("  - results/attribute_correlations.csv     : Correlation analysis")
        print("  - results/reliability_summary.csv        : Cronbach's alpha")
        print()
        print("✓ Experiment successfully replicated!")
        print()

    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Replicate the student performance prediction experiment'
    )
    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='Skip data generation step and use existing data'
    )

    args = parser.parse_args()

    run_pipeline(skip_data_generation=args.skip_data_generation)


if __name__ == "__main__":
    main()
