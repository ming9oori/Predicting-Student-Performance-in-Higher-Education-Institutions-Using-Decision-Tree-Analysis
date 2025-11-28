"""
Data preprocessing module following the paper's methodology.

This module implements:
1. Removing rows with empty values (8 rows)
2. Creating the 'Failed' column based on Q12 (Number of Failed Courses)
3. Data cleaning and conversion
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    Preprocessor for student performance data following the paper's approach.
    """

    def __init__(self, data_path):
        """
        Initialize preprocessor with raw data.

        Parameters:
        -----------
        data_path : str
            Path to the raw data CSV file
        """
        self.data_path = data_path
        self.df_raw = None
        self.df_clean = None

    def load_data(self):
        """Load raw data from CSV file"""
        print("Loading raw data...")
        self.df_raw = pd.read_csv(self.data_path)
        print(f"Raw data loaded: {len(self.df_raw)} rows, {len(self.df_raw.columns)} columns")
        return self

    def remove_missing_values(self):
        """
        Remove rows with missing values.
        Paper states: 8 rows with empty values were removed, leaving 151 answers.
        """
        print("\nRemoving rows with missing values...")
        initial_count = len(self.df_raw)

        # Remove rows with any missing values
        self.df_clean = self.df_raw.dropna()

        removed_count = initial_count - len(self.df_clean)
        print(f"Removed {removed_count} rows with missing values")
        print(f"Remaining rows: {len(self.df_clean)}")

        return self

    def create_failed_column(self):
        """
        Create 'Failed' column based on Q12 (Number of Failed Courses).

        According to the paper:
        If (Number of Failed Courses > 0), then Failed = 'F'
        Else Failed = 'P'
        F = Failed, P = Passed
        """
        print("\nCreating 'Failed' column...")

        def determine_failure(q12_value):
            """
            Determine if student failed based on Q12 value.

            Parameters:
            -----------
            q12_value : str
                Value from Q12 (Number of Failed Courses)

            Returns:
            --------
            str
                'F' for failed, 'P' for passed
            """
            if q12_value == '0':
                return 'P'
            else:
                return 'F'

        self.df_clean['Failed'] = self.df_clean['Q12'].apply(determine_failure)

        # Display distribution
        failed_count = (self.df_clean['Failed'] == 'F').sum()
        passed_count = (self.df_clean['Failed'] == 'P').sum()

        print(f"Class distribution:")
        print(f"  Passed (P): {passed_count} ({passed_count/len(self.df_clean)*100:.1f}%)")
        print(f"  Failed (F): {failed_count} ({failed_count/len(self.df_clean)*100:.1f}%)")

        return self

    def convert_categorical_to_numeric(self):
        """
        Convert categorical variables to numeric for analysis.
        This makes the data compatible with correlation analysis.
        """
        print("\nConverting categorical variables to numeric...")

        # Create a copy for numeric conversion
        self.df_numeric = self.df_clean.copy()

        # Convert all non-numeric columns to category codes
        for col in self.df_numeric.columns:
            if self.df_numeric[col].dtype == 'object':
                self.df_numeric[col] = pd.Categorical(self.df_numeric[col]).codes

        print("Conversion complete")

        return self

    def save_processed_data(self, output_path):
        """
        Save the processed data to CSV.

        Parameters:
        -----------
        output_path : str
            Path to save the processed data
        """
        print(f"\nSaving processed data to: {output_path}")
        self.df_clean.to_csv(output_path, index=False)

        # Also save numeric version
        numeric_path = output_path.replace('.csv', '_numeric.csv')
        print(f"Saving numeric data to: {numeric_path}")
        self.df_numeric.to_csv(numeric_path, index=False)

        return self

    def get_summary(self):
        """Print summary statistics of processed data"""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY")
        print("=" * 60)
        print(f"Final dataset shape: {self.df_clean.shape}")
        print(f"Total features: {len(self.df_clean.columns) - 1} (+ 1 target variable)")
        print(f"Target variable: Failed (F/P)")
        print()
        print("First 5 rows of processed data:")
        print(self.df_clean.head())
        print()
        print("Data types:")
        print(self.df_clean.dtypes.value_counts())

    def preprocess(self, output_path):
        """
        Execute complete preprocessing pipeline.

        Parameters:
        -----------
        output_path : str
            Path to save processed data

        Returns:
        --------
        pd.DataFrame
            Processed dataframe
        """
        self.load_data()
        self.remove_missing_values()
        self.create_failed_column()
        self.convert_categorical_to_numeric()
        self.save_processed_data(output_path)
        self.get_summary()

        return self.df_clean, self.df_numeric


def main():
    """Main preprocessing function"""
    from utils import get_data_dir

    print("=" * 60)
    print("STUDENT PERFORMANCE DATA PREPROCESSING")
    print("=" * 60)

    # Get data directory
    data_dir = get_data_dir()

    # Initialize preprocessor
    preprocessor = DataPreprocessor(str(data_dir / 'student_data_raw.csv'))

    # Run preprocessing pipeline
    df_clean, df_numeric = preprocessor.preprocess(str(data_dir / 'student_data_processed.csv'))

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
