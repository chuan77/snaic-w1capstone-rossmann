# Rossmann Sales Forecasting

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Predicting sales for Rossmann drugstores using machine learning models

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict future sales for Rossmann drugstores based on historical sales data, store information, and external factors. The solution implements multiple machine learning models to achieve accurate forecasting.

### Key Features
- Data preprocessing and feature engineering
- Multiple ML models (LightGBM, Random Forest, XGBoost)
- Comprehensive exploratory data analysis (EDA)
- Model evaluation and comparison
- Submission file generation for Kaggle competitions

## Project Structure

```
.
├── data/                      # Data files
│   ├── train.csv             # Training data
│   ├── test.csv              # Test data for predictions
│   ├── store.csv             # Store information
│   └── mechanical_failure_analysis.csv
├── outputs/                   # Generated outputs
│   ├── Plots and visualizations
│   ├── prediction submissions
│   └── analysis reports
├── submissions/               # Final submission files
│   └── Multiple model predictions (LightGBM, Random Forest)
├── .venv/                     # Python virtual environment
├── .git/                      # Git repository
├── .continue/                 # Continue IDE configuration
├── .vscode/                   # VS Code configuration
├── pyproject.toml            # Project dependencies and metadata
├── README.md                 # Project documentation
├── rossmann_final.py         # Main project script
├── part_d_failure_analysis.py # Analysis script
└── eda_JL_CPU.ipynb          # EDA notebook
```

## Data Description

### Training Data (`train.csv`)
- **Date**: Sale date
- **Store**: Store ID
- **Sales**: Sales amount
- **Customers**: Number of customers
- **Open**: Store open status
- **Promo**: Promo indicator
- **StateHoliday**: State holiday code
- **SchoolHoliday**: School holiday indicator

### Test Data (`test.csv`)
Same columns as training data except **Sales** (target variable to predict)

### Store Data (`store.csv`)
- **Store**: Store ID
- **StoreType**: Store type
- **Assortment**: Assortment level
- **CompetitionDistance**: Distance to nearest competitor
- **CompetitionOpenSince**: Year and month competitor opened
- **Promo2**: Promo2 indicator
- **Promo2Since**: Year and week promo2 started

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or uv package manager

### Setup Steps

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd rossmann-sales-forecasting
   ```

2. Create a virtual environment (optional but recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -e .
   # or using uv
   uv sync
   ```

## Usage

### Running the Main Project Script

Execute the main forecasting pipeline:
```bash
python rossmann_final.py
```

### Running the Analysis Script

Execute the failure analysis:
```bash
python part_d_failure_analysis.py
```

### Running the EDA Notebook

Open the exploratory data analysis notebook:
```bash
jupyter notebook eda_JL_CPU.ipynb
```

### Generating Predictions

The main script automatically:
1. Loads and preprocesses data
2. Trains models
3. Evaluates performance
4. Generates submission files in the `submissions/` directory

## Models

### LightGBM
- Fast and efficient gradient boosting framework
- Handles categorical features well
- Often produces excellent results with minimal tuning

### Random Forest
- Ensemble method using multiple decision trees
- Robust to overfitting
- Provides feature importance analysis

## Results

### Evaluation Metrics
- **RMSE** (Root Mean Square Error)

### Submission Files
Multiple submission files are generated for different model configurations:
- `rossmann_lgbm_business.csv`
- `rossmann_lgbm_kaggle.csv`
- `rossmann_lgbm_predictions_full.csv`
- `rossmann_rf_business.csv`
- `rossmann_rf_kaggle.csv`
- `rossmann_rf_predictions_full.csv`

### Visualization Outputs
The project generates various plots:
- Feature importance charts
- Error distribution analysis
- Actual vs. predicted sales scatter plots
- Failure modes visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Rossmann for providing the dataset
- ML community for inspiration and resources