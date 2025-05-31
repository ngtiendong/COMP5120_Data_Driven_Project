# Life Expectancy, Happiness, and Economic Analysis Dashboard

A comprehensive Shiny for Python application that analyzes the relationships between life expectancy, happiness scores, and various economic and environmental factors across countries and time periods.

## Purpose

This interactive dashboard provides a multi-dimensional analysis platform for understanding the complex relationships between human well-being, economic development, and environmental factors. The application enables researchers, policymakers, and data analysts to:

- **Explore Global Patterns**: Visualize life expectancy, happiness scores, and economic indicators across countries and time periods through interactive world maps
- **Analyze Temporal Trends**: Track how happiness, GDP, and life expectancy evolve over time for individual countries or entire continents
- **Compare Multiple Factors**: Examine how various economic, health, and environmental factors correlate with life expectancy outcomes
- **Identify Top Performers**: Analyze characteristics of countries with highest/lowest life expectancy to understand success factors
- **Predict Outcomes**: Use machine learning models to understand which factors most strongly predict life expectancy and happiness

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
# Navigate to your desired directory
cd /path/to/your/projects

# If using git, clone the repository
# git clone <repository-url>

# Or download and extract the project files
```

### Step 2: Install Required Dependencies

```bash
# Navigate to the project directory
cd life_expectancy_app

# Install required packages
pip install -r requirements.txt
```

The application requires the following packages:
- `shiny>=0.5.0` - Web application framework
- `pandas>=1.3.0` - Data manipulation and analysis
- `numpy>=1.20.0` - Numerical computing
- `matplotlib>=3.5.0` - Static plotting
- `plotly>=5.5.0` - Interactive visualizations
- `scikit-learn>=1.0.0` - Machine learning algorithms
- `seaborn>=0.11.0` - Statistical data visualization

### Step 3: Prepare the Data

The application comes with processed data, but if you need to regenerate it:

```bash
# Run the data processing script
python create_dataset.py
```

## How to Run the Application

### Local Development Server

```bash
# Start the Shiny application
shiny run app.py --reload

# Or for production deployment
shiny run app.py --host 0.0.0.0 --port 8000
```

The application will be available at:
- Local access: `http://localhost:8000`
- Network access: `http://your-ip-address:8000`

### Production Deployment

For production deployment, consider using:
- **ShinyProxy** for containerized deployment
- **Posit Connect** for enterprise deployment
- **Docker** for containerization
- **Nginx** as a reverse proxy for performance

## Data Creation and Processing Pipeline (`create_dataset.py`)

## Data Processing Pipeline (create_dataset.py)

### 1. Data Source Integration

The data processing pipeline integrates multiple datasets to create a comprehensive analytical foundation:

- **Primary Dataset**: `data/all.csv` - Contains comprehensive country-level indicators from World Bank and other sources including:
  - Life expectancy metrics (total, male, female)
  - Economic indicators (GDP, expenditures, account balances)
  - Health expenditure data (government and total)
  - Environmental metrics (CO2 emissions by sector)
  - Population demographics (age groups, gender distribution)
  - Geographic and development indicators

- **Secondary Dataset**: `data/happiness-cantril-ladder.csv` - Contains happiness/well-being scores (Cantril Ladder) from the World Happiness Report
- **Configuration Data**: `data/config/continents.json` - Mapping of countries to continents for regional analysis
- **Merge Strategy**: Left join on Country Name, Country Code, and Year to preserve all records from primary dataset

### 2. Data Cleaning and Filtering

```python
def merge_datasets():
    # Load datasets
    all_df = pd.read_csv('data/all.csv')
    happiness_df = pd.read_csv('data/happiness-cantril-ladder.csv')
```

The pipeline implements comprehensive data cleaning:

- **Temporal Filtering**: Removes data from 2024 to focus on complete historical records
- **Geographic Filtering**: Excludes 75+ regional aggregates, income groups, and non-sovereign entities including:
  - Regional groups (e.g., "Africa Eastern and Southern", "Arab World")
  - Income classifications (e.g., "High income", "Low & middle income")
  - Political entities (e.g., "European Union", "OECD members")
  - Small territories and dependencies
- **Quality Control**: Removes rows where all data values are missing

### 3. Column Standardization and Mapping

The pipeline standardizes column names for clarity and consistency:

```python
column_mapping = {
    'Country Name': 'Country',
    'Time': 'Year',
    'Life expectancy at birth, total (years)': 'Life Expectancy',
    'GDP (current US$)': 'GDP',
    'Cantril ladder score': 'Happiness Score',
    # ... additional mappings
}
```

Key transformations include:
- **Identifier Columns**: Country Name → Country, Time → Year, Country Code → Code
- **Life Expectancy**: Total, Male, and Female life expectancy at birth
- **Economic Indicators**: GDP, National Expenditure, Current Account Balance
- **Health Metrics**: Government and Total health expenditure as % of GDP
- **Environmental**: CO2 emissions from various sources (Agriculture, Energy, Industrial, Transport)
- **Demographics**: Population breakdowns by age groups and gender
- **Well-being**: Cantril Ladder Score → Happiness Score

### 4. Data Type Optimization and Derived Metrics

```python
# Convert to numeric with error handling
for col in merged_df.columns:
    if col not in exclude_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Calculate derived metrics
if 'GDP' in merged_df.columns and 'Population Total' in merged_df.columns:
    merged_df['GDP per Capita'] = merged_df['GDP'] / merged_df['Population Total']
```

### 5. Missing Value Handling Strategy

The pipeline implements sophisticated missing value handling:

- **Happiness Score 2013**: Interpolated using average of 2012 and 2014 values for continuity
- **Israel 1960-1965**: Historical gaps filled with column means for statistical completeness
- **Luxembourg Fertility Rate**: Filled with column mean to maintain demographic analysis
- **Zero-filling**: CO2 emissions, health expenditure, GDP metrics filled with 0 (missing = no activity/data)

### 6. Data Quality Optimization

```python
# Remove columns with limited analytical value
columns_to_drop = ['Deaths Non-Communicable', 'Deaths Communicable Diseases', 
                   'Iodized Salt Consumption', 'Low Birthweight', ...]
merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

# Save processed dataset
merged_df.to_csv('data/all_with_happiness.csv', index=False)
```

Final processing includes:
- Removal of columns with sparse data or limited analytical value
- Export to `data/all_with_happiness.csv` for application use
- Comprehensive logging of dataset transformations

## Application Structure and Tabs Analysis (`app.py`)

The application is built using Shiny for Python and features five comprehensive analysis tabs, each with specific data processing strategies and visualization approaches.

### Tab 1: World Map Overview

**Purpose**: Global visualization and country-specific exploration

**Data Processing Strategy**:
```python
def mean_positive(series):
    """Calculate mean excluding zero values to prevent data quality issues"""
    positive_values = series[series > 0]
    return positive_values.mean() if len(positive_values) > 0 else 0

# Temporal aggregation for selected year range
avg_numeric = year_data.groupby('Country')[numeric_cols].agg(mean_positive)
```

Key processing features:
- **Temporal Aggregation**: Calculates averages over user-selected year ranges using `mean_positive()` function
- **Zero Handling**: Excludes zero values from averaging to prevent data quality issues
- **Fallback Logic**: Uses most recent available year if aggregation fails
- **Color Mapping**: Dynamic choropleth coloring based on Life Expectancy, Happiness Score, or GDP per Capita

**Interactive Features**:
- Interactive year range selection with averaging option
- Multiple color scales (Viridis, Plasma, Blues, Reds, Greens)
- Country-specific detail panel with key metrics
- Hover data showing Life Expectancy, Happiness Score, and GDP per Capita
- Real-time country selection and statistics display

### Tab 2: Happiness, GDP, and Life Expectancy Trends

**Purpose**: Time series analysis for individual countries or continental aggregates

**Data Processing Strategy**:
```python
if view_type == "Country View":
    # Individual country analysis
    filtered_data = data[(data['Country'] == selected_location) & 
                        (data['Year'] >= year_min) & 
                        (data['Year'] <= year_max)]
else:
    # Continental aggregation
    continent_countries = continents_data[selected_location]
    continent_data = data[data['Country'].isin(continent_countries)]
    
    # Calculate continent averages
    agg_dict = {col: 'mean' for col in available_factor_cols}
    plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
```

Key processing features:
- **Dual View System**: Switch between country-level analysis and continental aggregation
- **Continental Aggregation**: Groups countries by continent using `continents.json` mapping
- **Missing Value Handling**: Replaces zeros with NaN for cleaner visualizations
- **Multi-panel Display**: Three separate subplots for each metric with proper scaling

**Visualization Features**:
- Dynamic selector switching between country and continent views
- Three-panel subplot showing Life Expectancy (years), GDP per Capita (USD), and Happiness Score (0-10 scale)
- Year range filtering for focused analysis
- Proper axis labeling and scaling for each metric
- Color-coded trend lines for easy comparison

### Tab 3: Factors Indicator

**Purpose**: Multi-factor time series analysis with life expectancy as reference

**Data Processing Strategy**:
```python
# Normalization for comparative visualization
if factor == 'Happiness Score':
    # Scale happiness score (0-10) to life expectancy range
    normalized_values = plot_data_viz[factor] * (valid_life_exp_values.max() / 10)
elif factor == 'GDP per Capita':
    # Scale GDP to life expectancy range
    max_factor = valid_factor_values.max()
    if max_factor > 0:
        normalized_values = (plot_data_viz[factor] / max_factor) * valid_life_exp_values.max()
else:
    # General normalization for other factors
    normalized_values = (plot_data_viz[factor] / max_factor) * valid_life_exp_values.max()
```

Key processing features:
- **Normalization Strategy**: Scales all factors to life expectancy range for comparative visualization
- **Special Scaling**: Happiness Score (0-10 scale) and GDP factors get custom scaling algorithms
- **Reference Line**: Life Expectancy always shown as red reference line for comparison
- **Factor Selection**: User-selectable subset of available economic and environmental factors

**Advanced Features**:
- Normalized multi-line time series plots
- Life expectancy as constant reference for comparison
- Color-coded factor lines with comprehensive legend
- Spider chart visualization for factor impact analysis
- Continental vs. country analysis options

### Tab 4: Factors Indicator by Top Countries

**Purpose**: Comparative analysis focusing on highest/lowest performing countries

**Data Processing Strategy**:
```python
@reactive.Calc
def get_ranked_countries():
    # Calculate ranking based on average life expectancy
    year_min, year_max = input.years_range4()
    rank_data = data[(data['Year'] >= year_min) & (data['Year'] <= year_max)]
    
    # Calculate country averages and rank
    country_avg = rank_data.groupby('Country')['Life Expectancy'].mean().reset_index()
    country_avg = country_avg.sort_values('Life Expectancy', 
                                        ascending=(input.rank_type() == "Lowest Life Expectancy"))
```

Key processing features:
- **Ranking Algorithm**: Calculates average life expectancy over selected periods
- **Top/Bottom Selection**: Dynamic country selection based on life expectancy performance
- **Correlation Analysis**: Comprehensive statistical analysis with zero-value exclusion
- **Multi-plot Generation**: Heatmaps, pair plots, and distribution plots using seaborn

**Statistical Analysis Features**:
- Dynamic country ranking (5, 10, 15, 20 countries)
- Correlation matrix with statistical significance testing
- Pair plots for multi-dimensional relationship analysis
- Distribution plots for factor understanding
- Data quality reporting (sample sizes, exclusions)

**Statistical Methods**:
- **Correlation Matrix**: Lower triangle masked heatmap with color coding
- **Pair Plots**: Scatter plots with regression lines and correlation coefficients
- **Distribution Analysis**: Histograms with KDE for factor distribution understanding

### Tab 5: Predictive Analysis using ML

**Purpose**: Machine learning-based prediction and feature importance analysis

**Data Processing Strategy**:
```python
@reactive.Calc
def regression_model():
    # Dual target analysis
    targets = ['Life Expectancy', 'Happiness Score']
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Train-test split and model evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature scaling for linear models
    if model_name == 'Linear Regression':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
```

Key processing features:
- **Dual Target Analysis**: Separate models for Life Expectancy and Happiness Score prediction
- **Model Comparison**: Tests Linear Regression, Random Forest, and Gradient Boosting
- **Train-Test Split**: 80-20 split with proper evaluation metrics (R², RMSE, MAE)
- **Feature Scaling**: StandardScaler for linear models, raw features for tree-based models
- **Best Model Selection**: Automatic selection based on test R² scores

**Machine Learning Features**:
- **Prediction Plots**: Actual vs. Predicted scatter plots with perfect prediction reference lines
- **Feature Importance**: Grouped bar charts comparing factor importance for both targets
- **Model Performance**: Summary statistics and best model identification with overfitting detection
- **Correlation Analysis**: Pearson and Spearman correlations with significance testing

**Advanced Analytics**:
- Cross-validation approach with multiple algorithms
- Feature importance extraction (coefficients for linear, importances for tree-based)
- Statistical significance testing (p-values for correlations)
- Performance metrics comparison table
- Policy insights through positive/negative impact identification

## Technical Architecture

### Core Dependencies and Framework
```python
from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

### Data Loading and Preprocessing
```python
# Load processed dataset
data = pd.read_csv('data/all_with_happiness.csv')

# Generate dynamic factor choices
exclude_columns = ['Country', 'Year', 'Code', 'Year_Code', 'Life Expectancy', 
                  'Life Expectancy Male', 'Life Expectancy Female', 'Happiness Score']
factor_choices = [col for col in data.columns.tolist() if col not in exclude_columns]

# Load continent mapping
with open('data/config/continents.json', 'r') as f:
    continents_data = json.load(f)
```

### Error Handling and Validation
- **Graceful Degradation**: Fallback visualizations when data is insufficient
- **User Feedback**: Clear error messages and loading states
- **Data Validation**: Input validation and range checking
- **Missing Data**: Robust handling of missing values and zero values

### Performance Optimization
- **Reactive Computing**: Calculations triggered only on user input changes
- **Data Caching**: Efficient data filtering and aggregation using pandas
- **Memory Management**: Proper cleanup of matplotlib figures to prevent memory leaks
- **Lazy Loading**: Components rendered only when needed

### File Structure
```
life_expectancy_app/
├── app.py                          # Main Shiny application
├── create_dataset.py               # Data processing pipeline
├── requirements.txt                # Python dependencies
├── README.md                       # Documentation
├── data/
│   ├── all.csv                     # Raw primary dataset
│   ├── happiness-cantril-ladder.csv # Raw happiness data
│   ├── all_with_happiness.csv      # Processed final dataset
│   └── config/
│       └── continents.json         # Country-continent mapping
└── www/
    └── styles.css                  # Custom CSS styling
```

This comprehensive dashboard provides researchers and policymakers with powerful tools to understand the complex relationships between economic development, environmental factors, and human well-being across different countries and time periods.
