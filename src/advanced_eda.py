"""
Advanced Exploratory Data Analysis (EDA) for Presentation

This module provides comprehensive EDA including:
- Distribution analysis
- Correlation heatmaps
- Feature relationships
- Success/Failure comparisons
- Statistical insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils import get_data_dir, get_results_dir
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedEDA:
    """Comprehensive EDA for student performance data"""

    def __init__(self):
        """Initialize EDA with data loading"""
        data_dir = get_data_dir()
        self.results_dir = get_results_dir()

        # Load data
        self.df = pd.read_csv(data_dir / 'student_data_processed.csv')
        self.df_numeric = pd.read_csv(data_dir / 'student_data_processed_numeric.csv')

        print(f"Data loaded: {self.df.shape}")
        print(f"Target distribution:\n{self.df['Failed'].value_counts()}")

    def demographic_analysis(self):
        """Analyze demographic distributions"""
        print("\n" + "="*80)
        print("DEMOGRAPHIC ANALYSIS")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Q1: Department
        ax = axes[0, 0]
        dept_counts = self.df.groupby(['Q1', 'Failed']).size().unstack(fill_value=0)
        dept_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Department vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Department')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])
        ax.tick_params(axis='x', rotation=0)

        # Q2: Age
        ax = axes[0, 1]
        age_counts = self.df.groupby(['Q2', 'Failed']).size().unstack(fill_value=0)
        age_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Age vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q3: Stage (Year)
        ax = axes[0, 2]
        stage_counts = self.df.groupby(['Q3', 'Failed']).size().unstack(fill_value=0)
        stage_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Academic Year vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q4: Gender
        ax = axes[1, 0]
        gender_counts = self.df.groupby(['Q4', 'Failed']).size().unstack(fill_value=0)
        gender_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Gender vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Gender')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])
        ax.tick_params(axis='x', rotation=0)

        # Q6: Marital Status
        ax = axes[1, 1]
        marital_counts = self.df.groupby(['Q6', 'Failed']).size().unstack(fill_value=0)
        marital_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Marital Status vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Status')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q7: Working Status
        ax = axes[1, 2]
        work_counts = self.df.groupby(['Q7', 'Failed']).size().unstack(fill_value=0)
        work_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Working Status vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Working')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])
        ax.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'eda_demographics.png', dpi=300, bbox_inches='tight')
        print("✓ Demographic analysis saved")

        return fig

    def academic_analysis(self):
        """Analyze academic performance factors"""
        print("\n" + "="*80)
        print("ACADEMIC FACTORS ANALYSIS")
        print("="*80)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Q12: Number of Failed Courses
        ax = axes[0, 0]
        fail_counts = self.df['Q12'].value_counts().sort_index()
        fail_counts.plot(kind='bar', ax=ax, color='coral', alpha=0.7)
        ax.set_title('Distribution of Failed Courses', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Failed Courses')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=0)

        # Q13: Absence Days
        ax = axes[0, 1]
        absence_counts = self.df.groupby(['Q13', 'Failed']).size().unstack(fill_value=0)
        absence_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Absence Days vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Absence Days')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q14: Number of Credits
        ax = axes[0, 2]
        credits_counts = self.df.groupby(['Q14', 'Failed']).size().unstack(fill_value=0)
        credits_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Credits vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Credits')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q15: GPA
        ax = axes[1, 0]
        gpa_counts = self.df.groupby(['Q15', 'Failed']).size().unstack(fill_value=0)
        gpa_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('GPA vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('GPA Range')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q16: Completed Credits
        ax = axes[1, 1]
        comp_credits = self.df.groupby(['Q16', 'Failed']).size().unstack(fill_value=0)
        comp_credits.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Completed Credits vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Completed Credits')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])

        # Q17: Years of Study
        ax = axes[1, 2]
        years_counts = self.df.groupby(['Q17', 'Failed']).size().unstack(fill_value=0)
        years_counts.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)
        ax.set_title('Years of Study vs Performance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Years')
        ax.set_ylabel('Count')
        ax.legend(['Passed', 'Failed'])
        ax.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'eda_academic.png', dpi=300, bbox_inches='tight')
        print("✓ Academic analysis saved")

        return fig

    def likert_scale_analysis(self):
        """Analyze Likert scale questions (Q18-Q60)"""
        print("\n" + "="*80)
        print("LIKERT SCALE ANALYSIS (Study Skills, Motivation, etc.)")
        print("="*80)

        # Group questions by category
        categories = {
            'Study Skills': [f'Q{i}' for i in range(18, 22)],
            'Motivation': [f'Q{i}' for i in range(22, 27)],
            'Relationships': [f'Q{i}' for i in range(27, 31)],
            'Health': [f'Q{i}' for i in range(31, 35)],
            'Time Mgmt': [f'Q{i}' for i in range(35, 40)],
            'Money Mgmt': [f'Q{i}' for i in range(40, 45)],
            'Purpose': [f'Q{i}' for i in range(45, 50)],
            'Career': [f'Q{i}' for i in range(50, 54)],
            'Resources': [f'Q{i}' for i in range(54, 57)],
            'Self-Esteem': [f'Q{i}' for i in range(57, 61)]
        }

        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes = axes.flatten()

        for idx, (cat_name, questions) in enumerate(categories.items()):
            ax = axes[idx]

            # Calculate mean scores for passed vs failed students
            passed_scores = self.df_numeric[self.df_numeric['Failed'] == 0][questions].mean().mean()
            failed_scores = self.df_numeric[self.df_numeric['Failed'] == 1][questions].mean().mean()

            bars = ax.bar(['Passed', 'Failed'], [passed_scores, failed_scores],
                         color=['green', 'red'], alpha=0.7)
            ax.set_title(cat_name, fontsize=12, fontweight='bold')
            ax.set_ylabel('Average Score (1-5)')
            ax.set_ylim([1, 5])
            ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'eda_likert_categories.png', dpi=300, bbox_inches='tight')
        print("✓ Likert scale analysis saved")

        return fig

    def correlation_heatmap(self):
        """Create detailed correlation heatmap"""
        print("\n" + "="*80)
        print("CORRELATION HEATMAP")
        print("="*80)

        # Top 30 most correlated features
        correlations = pd.read_csv(self.results_dir / 'attribute_correlations.csv')
        top_features = correlations.head(30)['Attribute'].tolist()

        # Add target variable
        top_features_with_target = top_features + ['Failed']

        # Calculate correlation matrix
        corr_matrix = self.df_numeric[top_features_with_target].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(20, 18))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, square=True,
                   linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax)

        ax.set_title('Correlation Heatmap - Top 30 Features',
                    fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved")

        return fig

    def statistical_tests(self):
        """Perform statistical tests to find significant differences"""
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)

        # Separate passed and failed students
        passed = self.df_numeric[self.df_numeric['Failed'] == 0]
        failed = self.df_numeric[self.df_numeric['Failed'] == 1]

        # Likert scale questions
        likert_questions = [f'Q{i}' for i in range(18, 61)]

        results = []

        for q in likert_questions:
            # T-test
            t_stat, p_value = stats.ttest_ind(passed[q], failed[q])

            # Effect size (Cohen's d)
            mean_diff = passed[q].mean() - failed[q].mean()
            pooled_std = np.sqrt((passed[q].std()**2 + failed[q].std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            results.append({
                'Question': q,
                'Passed_Mean': passed[q].mean(),
                'Failed_Mean': failed[q].mean(),
                'Mean_Diff': mean_diff,
                'T_Statistic': t_stat,
                'P_Value': p_value,
                'Cohens_D': cohens_d,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('P_Value')

        # Save results
        results_df.to_csv(self.results_dir / 'statistical_tests.csv', index=False)

        # Print top 10 most significant
        print("\nTop 10 Most Significant Differences (p < 0.05):")
        print(results_df.head(10)[['Question', 'Passed_Mean', 'Failed_Mean',
                                    'Mean_Diff', 'P_Value', 'Cohens_D']].to_string())

        # Visualize
        fig, ax = plt.subplots(figsize=(14, 8))

        top_20 = results_df.head(20)

        x = np.arange(len(top_20))
        width = 0.35

        bars1 = ax.bar(x - width/2, top_20['Passed_Mean'], width,
                      label='Passed', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, top_20['Failed_Mean'], width,
                      label='Failed', color='red', alpha=0.7)

        ax.set_xlabel('Question', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Score', fontsize=12, fontweight='bold')
        ax.set_title('Top 20 Questions with Significant Differences\n(Passed vs Failed Students)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_20['Question'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.results_dir / 'eda_statistical_differences.png', dpi=300, bbox_inches='tight')
        print("✓ Statistical tests saved")

        return results_df

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING EDA SUMMARY REPORT")
        print("="*80)

        report = []
        report.append("# Exploratory Data Analysis - Summary Report\n")
        report.append("="*80 + "\n\n")

        # Dataset overview
        report.append("## 1. Dataset Overview\n")
        report.append(f"- Total Students: {len(self.df)}\n")
        report.append(f"- Total Features: {len(self.df.columns) - 1}\n")
        report.append(f"- Passed Students: {(self.df['Failed'] == 'P').sum()} ({(self.df['Failed'] == 'P').sum()/len(self.df)*100:.1f}%)\n")
        report.append(f"- Failed Students: {(self.df['Failed'] == 'F').sum()} ({(self.df['Failed'] == 'F').sum()/len(self.df)*100:.1f}%)\n\n")

        # Key findings
        report.append("## 2. Key Findings from EDA\n\n")

        # Load statistical tests
        stats_df = pd.read_csv(self.results_dir / 'statistical_tests.csv')
        significant = stats_df[stats_df['Significant'] == 'Yes']

        report.append(f"### Statistical Significance\n")
        report.append(f"- {len(significant)} out of {len(stats_df)} questions show significant differences (p < 0.05)\n")
        report.append(f"- Top 3 most discriminative features:\n")
        for idx, row in stats_df.head(3).iterrows():
            report.append(f"  - {row['Question']}: Mean diff = {row['Mean_Diff']:.3f}, p = {row['P_Value']:.4f}\n")

        report.append("\n## 3. Recommendations for Intervention\n\n")
        report.append("Based on the analysis, focus on improving:\n")
        report.append("1. Student motivation and engagement\n")
        report.append("2. Time management skills\n")
        report.append("3. Study habits and techniques\n")
        report.append("4. Self-confidence and self-esteem\n")

        # Save report
        with open(self.results_dir / 'eda_summary_report.txt', 'w') as f:
            f.writelines(report)

        print("✓ Summary report saved")
        print("\n".join(report))


def main():
    """Run comprehensive EDA"""
    print("="*80)
    print("ADVANCED EXPLORATORY DATA ANALYSIS FOR PRESENTATION")
    print("="*80)

    eda = AdvancedEDA()

    # Run all analyses
    eda.demographic_analysis()
    eda.academic_analysis()
    eda.likert_scale_analysis()
    eda.correlation_heatmap()
    eda.statistical_tests()
    eda.generate_summary_report()

    print("\n" + "="*80)
    print("✓ ALL EDA COMPLETED!")
    print("="*80)
    print("\nGenerated files:")
    print("  - eda_demographics.png")
    print("  - eda_academic.png")
    print("  - eda_likert_categories.png")
    print("  - eda_correlation_heatmap.png")
    print("  - eda_statistical_differences.png")
    print("  - statistical_tests.csv")
    print("  - eda_summary_report.txt")


if __name__ == "__main__":
    main()
