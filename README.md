# Synthetic Data Generation Through Modeling & Simulation Techniques

## Overview

This project demonstrates the generation of synthetic datasets through discrete-event simulation using the SimPy framework, followed by comprehensive machine learning model evaluation. The approach enables controlled experimentation with various ML algorithms on simulated data, providing insights into model performance characteristics without relying on real-world data collection.

## Features

-  **Automated Simulation Pipeline**: Generate synthetic datasets through 1000+ simulation iterations
-  **Multi-Model Comparison**: Evaluate 5 different ML algorithms simultaneously
-  **Comprehensive Visualization**: Built-in plotting for data distributions and performance metrics
-  **Statistical Analysis**: Detailed metrics including R², MSE, MAE, and RMSE
-  **Modular Architecture**: Easy to extend with additional models or simulation parameters
-  **Detailed Logging**: Track simulation progress and model training phases

##  Methodology

### 1. Data Generation Phase
The simulation framework utilizes SimPy to create synthetic datasets through discrete-event modeling:
- **Simulation Runs**: 1000 independent iterations
- **Data Collection**: Systematic sampling of simulation states
- **Feature Engineering**: Extraction of relevant features from simulation events
- **Data Validation**: Quality checks and outlier detection

### 2. Preprocessing Phase
- Feature scaling and normalization
- Train-test split (80-20 ratio)
- Cross-validation setup
- Missing value handling (if applicable)

### 3. Model Training & Evaluation
Each algorithm is trained and evaluated using consistent metrics:
- Training phase with hyperparameter optimization
- Cross-validation for robust performance estimation
- Test set evaluation for final metrics
- Comparative analysis across all models

##  Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/synthetic-data-ml-simulation.git
cd synthetic-data-ml-simulation
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements File
Create a `requirements.txt` with the following dependencies:
```
simpy>=4.0.1
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

##  Usage

### Basic Usage

```python
# Run the complete simulation and analysis pipeline
python main.py
```

### Advanced Configuration

```python
# Customize simulation parameters
python main.py --num_simulations 2000 --random_seed 42

# Select specific models
python main.py --models lr dt rf

# Enable verbose logging
python main.py --verbose
```

### Code Example

```python
from simulation import DataSimulator
from models import MLModelComparison

# Initialize simulator
simulator = DataSimulator(num_runs=1000)

# Generate synthetic data
data = simulator.run_simulation()

# Train and evaluate models
comparison = MLModelComparison(data)
results = comparison.evaluate_all_models()

# Generate visualizations
comparison.plot_results()
```

##  Project Structure

```
synthetic-data-ml-simulation/
│
├── data/
│   ├── raw/                    # Raw simulation outputs
│   └── processed/              # Cleaned and preprocessed data
│
├── src/
│   ├── simulation.py           # SimPy simulation logic
│   ├── models.py               # ML model implementations
│   ├── preprocessing.py        # Data preprocessing utilities
│   ├── visualization.py        # Plotting functions
│   └── utils.py                # Helper functions
│
├── notebooks/
│   ├── exploration.ipynb       # Data exploration
│   └── analysis.ipynb          # Detailed analysis
│
├── results/
│   ├── figures/                # Generated plots
│   └── metrics/                # Performance metrics
│
├── tests/
│   ├── test_simulation.py      # Unit tests
│   └── test_models.py          # Model tests
│
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # License information
```

##  Machine Learning Models

### 1. Linear Regression
- **Type**: Parametric regression
- **Use Case**: Baseline model for linear relationships
- **Advantages**: Fast, interpretable, low variance
- **Limitations**: Assumes linear relationships

### 2. Decision Tree
- **Type**: Non-parametric regression/classification
- **Use Case**: Capturing non-linear patterns
- **Advantages**: Interpretable, handles non-linearity
- **Limitations**: Prone to overfitting

### 3. Random Forest
- **Type**: Ensemble method
- **Use Case**: Robust prediction with reduced overfitting
- **Advantages**: High accuracy, handles outliers
- **Limitations**: Less interpretable, computationally intensive

### 4. Support Vector Machine (SVM)
- **Type**: Kernel-based method
- **Use Case**: Complex decision boundaries
- **Advantages**: Effective in high dimensions
- **Limitations**: Sensitive to parameter tuning

### 5. K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Use Case**: Local pattern recognition
- **Advantages**: Simple, no training phase
- **Limitations**: Computationally expensive at prediction time

##  Results & Analysis

### Generated Dataset
The simulation produces a comprehensive dataset with multiple features representing various system states and events:

<img width="1400" height="600" alt="simulation_dataset_reordered" src="https://github.com/user-attachments/assets/bcb289dc-3dd3-4735-8a54-cf98bafd3bfd" />

**Dataset Characteristics:**
- Total samples: 1000 (from simulation runs)
- Features: Multiple engineered features from simulation states
- Target variable: Derived from simulation outcomes
- Data quality: No missing values, normalized distributions

### Performance Metrics

<img width="1200" height="400" alt="ml_model_comparison_reordered" src="https://github.com/user-attachments/assets/8e7e9a0d-0b8f-4b35-84bd-d5c8f9fc5a8f" />

**Evaluation Metrics Explained:**
- **R² Score**: Proportion of variance explained (closer to 1 is better)
- **MSE**: Mean Squared Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **RMSE**: Root Mean Squared Error (lower is better)

### Key Findings
1. **Best Overall Performance**: [Model name based on results]
2. **Training Efficiency**: [Fastest model]
3. **Prediction Accuracy**: [Most accurate model]
4. **Trade-offs**: Balance between accuracy and computational cost

##  Dependencies

### Core Libraries
- **SimPy**: Discrete-event simulation framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms

### Visualization
- **Matplotlib**: Static plotting
- **Seaborn**: Statistical data visualization

### Utilities
- **SciPy**: Scientific computing tools

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)


