# Predicting Student Performance in Higher Education Institutions Using Decision Tree Analysis

## Paper Replication Project

This project replicates the experiments from the paper:

**"Predicting Student Performance in Higher Education Institutions Using Decision Tree Analysis"**
Authors: Alaa Khalaf Hamoud, Ali Salah Hashim, Wid Aqeel Awadh
Published: February 2018
DOI: [10.9781/ijimai.2018.02.004](https://doi.org/10.9781/ijimai.2018.02.004)

---

## Table of Contents

- [Overview](#overview)
- [Paper Summary](#paper-summary)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Comparison with Paper](#comparison-with-paper)

---

## Overview

This project implements a complete replication of the student performance prediction study using three decision tree algorithms:

1. **J48** (C4.5) - Information gain-based classifier
2. **Random Tree** - Random split-based classifier
3. **REPTree** - Reduced Error Pruning Tree

The implementation follows the exact methodology described in the paper, including:
- Data preprocessing (removing missing values, creating target variable)
- Reliability analysis using Cronbach's Alpha
- Attribute selection using correlation analysis
- Model training with 10-fold cross-validation
- Comprehensive result visualization

---

## Paper Summary

### Objective
Predict student success/failure in higher education institutions to identify factors affecting academic performance.

### Dataset
- **161 questionnaires** collected from students
- **60 questions** covering:
  - Demographic information
  - Social information
  - Academic performance
  - Study skills
  - Motivation
  - Personal relationships
  - Health
  - Time management
  - Money management
  - Career planning
  - Self-esteem

### Key Findings (Paper)
- **Best Algorithm**: J48 achieved the best performance
- **Attribute Selection**: Removing the 20 least correlated attributes improved accuracy
- **Important Factors**: GPA, Credits, Study Skills, Motivation
- **Reliability**: Cronbach's Alpha = 0.85 (Good internal consistency)

---

## Project Structure

```
.
├── data/                           # Data files
│   ├── student_data_raw.csv       # Raw generated data (161 samples)
│   ├── student_data_processed.csv # Processed data (151 samples)
│   ├── student_data_processed_numeric.csv
│   └── student_data_filtered.csv  # Filtered data (top 40 attributes)
│
├── src/                           # Source code
│   ├── generate_dataset.py       # Dataset generation
│   ├── preprocess_data.py        # Data preprocessing
│   ├── reliability_analysis.py   # Cronbach's alpha calculation
│   ├── attribute_selection.py    # Correlation analysis
│   ├── decision_tree_models.py   # Model implementation
│   └── visualize_results.py      # Visualization and reporting
│
├── models/                        # Trained models
│   ├── J48_model.pkl
│   ├── RandomTree_model.pkl
│   └── REPTree_model.pkl
│
├── results/                       # Analysis results
│   ├── model_comparison_full.csv
│   ├── model_comparison_filtered.csv
│   ├── attribute_correlations.csv
│   ├── reliability_summary.csv
│   ├── figure2_performance_comparison.png
│   ├── figure3_j48_tree.png
│   └── j48_feature_importance.png
│
├── notebooks/                     # Jupyter notebooks
│   └── student_performance_analysis.ipynb
│
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

The required packages include:
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- scipy >= 1.10.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- jupyter >= 1.0.0

---

## Quick Start

### Run the complete pipeline:

```bash
python main.py
```

This will execute all steps:
1. Generate sample dataset (161 responses)
2. Preprocess data (remove missing values → 151 responses)
3. Calculate reliability (Cronbach's alpha)
4. Select attributes (correlation analysis)
5. Train and evaluate models (10-fold CV)
6. Generate visualizations and reports

### Expected Runtime
- Complete pipeline: ~2-5 minutes
- Results will be saved to `data/`, `models/`, and `results/` directories

---

## Detailed Usage

### 1. Generate Dataset

```bash
cd src
python generate_dataset.py
```

Generates synthetic student data matching the paper's questionnaire structure.

### 2. Preprocess Data

```bash
python preprocess_data.py
```

- Removes 8 rows with missing values (161 → 151 samples)
- Creates `Failed` column: `If (Q12 > 0) then 'F' else 'P'`
- Converts categorical to numeric

### 3. Reliability Analysis

```bash
python reliability_analysis.py
```

Calculates Cronbach's Alpha for internal consistency.

### 4. Attribute Selection

```bash
python attribute_selection.py
```

- Calculates Pearson correlation for each attribute
- Ranks attributes by correlation strength
- Selects top 40 attributes (removes bottom 20)

### 5. Train Models

```bash
python decision_tree_models.py
```

Trains and evaluates three classifiers:
- J48 (C4.5)
- Random Tree
- REPTree

Uses 10-fold stratified cross-validation.

### 6. Visualize Results

```bash
python visualize_results.py
```

Generates:
- Performance comparison charts
- Decision tree visualizations
- Feature importance plots
- Comprehensive summary report

### 7. Interactive Analysis

```bash
jupyter notebook notebooks/student_performance_analysis.ipynb
```

Provides interactive analysis with all visualizations and comparisons.

---

## Methodology

### 1. Data Collection
- **161 questionnaires** with 60 questions
- Questions cover demographic, social, academic, and personal factors

### 2. Data Preprocessing
- Remove rows with missing values (8 rows removed)
- Create binary target variable `Failed` based on number of failed courses
- Convert categorical variables to numeric codes

### 3. Reliability Analysis
- Calculate **Cronbach's Alpha** to measure internal consistency
- Target: α ≥ 0.7 (acceptable reliability)

### 4. Attribute Selection
- Use **CorrelationAttributeEval** (Pearson correlation)
- Rank attributes by correlation with target variable
- Remove 20 least correlated attributes

### 5. Model Training
Three decision tree algorithms:

| Algorithm | Implementation Details |
|-----------|----------------------|
| **J48** | criterion='entropy', splitter='best' |
| **Random Tree** | criterion='gini', splitter='random' |
| **REPTree** | criterion='entropy', ccp_alpha for pruning |

### 6. Evaluation
- **10-fold stratified cross-validation**
- Metrics: TP Rate, FP Rate, Precision, Recall, Accuracy, F1-Score

---

## Results

### Model Performance (With Attribute Filter)

Based on the paper's Table V, expected results:

| Classifier | TP Rate | FP Rate | Precision | Recall |
|------------|---------|---------|-----------|--------|
| **J48** | 0.634 | 0.409 | 0.629 | 0.634 |
| Random Tree | 0.614 | 0.423 | 0.597 | 0.614 |
| REPTree | 0.601 | 0.488 | 0.583 | 0.601 |

**Best Model**: J48 (C4.5)

### Reliability Analysis

- **Cronbach's Alpha**: 0.85 (Good internal consistency)
- Number of items: 60
- Number of respondents: 161

### Top Correlated Attributes

According to the paper's Table III, most important factors:
1. Q14 - Number of Credits
2. Q25 - Motivation (excitement about courses)
3. Q60 - Self-esteem (ability to do college work)
4. Q15 - GPA
5. Q46 - Personal purpose (responsibility for education)

---

## Key Findings

### 1. Best Algorithm
**J48 (C4.5)** outperformed Random Tree and REPTree in terms of:
- True Positive Rate
- Precision
- Overall accuracy

### 2. Impact of Attribute Selection
Removing the 20 least correlated attributes:
- ✓ Improved model performance
- ✓ Reduced complexity
- ✓ Better generalization

### 3. Most Important Factors

**High Impact on Student Success:**
- Academic factors (GPA, Credits)
- Motivation and study skills
- Time management
- Personal purpose and responsibility

**Low Impact:**
- Demographics (Age, Gender)
- Marital status
- Some social factors

### 4. Practical Applications

The model can help:
- **Students**: Identify areas for improvement
- **Instructors**: Provide targeted support
- **Administrators**: Improve academic programs
- **Early Warning**: Detect at-risk students

---

## Comparison with Paper

### Similarities
✓ Same methodology (data preprocessing, attribute selection, 10-fold CV)
✓ Same algorithms (J48, Random Tree, REPTree)
✓ J48 identified as best performing algorithm
✓ Attribute filtering improved performance
✓ Similar correlation patterns observed

### Differences
- We use **synthetic data** (paper used real student surveys)
- Exact performance metrics may vary due to different data distributions
- Paper used Weka 3.8; we use Python scikit-learn

### Validation
Our implementation successfully replicates:
- ✓ Data preprocessing pipeline
- ✓ Reliability analysis (Cronbach's alpha)
- ✓ Attribute selection methodology
- ✓ Model training and evaluation approach
- ✓ Result visualization and interpretation

---

## Files Generated

### Data Files
- `student_data_raw.csv` - Raw survey data (161 samples)
- `student_data_processed.csv` - Cleaned data (151 samples)
- `student_data_filtered.csv` - Top 40 attributes

### Model Files
- `J48_model.pkl` - Trained J48 classifier
- `RandomTree_model.pkl` - Trained Random Tree classifier
- `REPTree_model.pkl` - Trained REPTree classifier

### Result Files
- `model_comparison_full.csv` - Performance without filtering
- `model_comparison_filtered.csv` - Performance with filtering
- `attribute_correlations.csv` - Correlation rankings
- `reliability_summary.csv` - Cronbach's alpha results
- `figure2_performance_comparison.png` - Performance charts
- `figure3_j48_tree.png` - Decision tree visualization
- `j48_feature_importance.png` - Feature importance chart

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{hamoud2018predicting,
  title={Predicting Student Performance in Higher Education Institutions Using Decision Tree Analysis},
  author={Hamoud, Alaa Khalaf and Hashim, Ali Salah and Awadh, Wid Aqeel},
  journal={International Journal of Interactive Multimedia and Artificial Intelligence},
  volume={5},
  number={2},
  pages={26--31},
  year={2018},
  doi={10.9781/ijimai.2018.02.004}
}
```

---

## Acknowledgments

- Original paper authors: Alaa Khalaf Hamoud, Ali Salah Hashim, Wid Aqeel Awadh
- University of Basra, Iraq
- Published in International Journal of Interactive Multimedia and Artificial Intelligence

---

## Contact

For questions or issues:
- Create an issue in this repository
- Review the original paper for methodology details

---

## License

This project is for educational and research purposes. Please refer to the original paper for research use.

---

**Note**: This implementation uses synthetic data generated to match the structure described in the paper. For production use with real student data, ensure proper data privacy and ethical considerations.
