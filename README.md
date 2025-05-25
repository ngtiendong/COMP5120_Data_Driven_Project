# Life Expectancy and Happiness Visualization

This Shiny application visualizes the relationship between life expectancy, happiness scores, and economic indicators across different countries.

## Features

- **Life Expectancy Tab**: Explore life expectancy trends across different countries over time
- **Happiness Tab**: Analyze happiness scores and their relationship with other factors
- **Economic Indicators Tab**: Investigate how economic factors correlate with life expectancy
- **Comparative Analysis Tab**: Compare multiple countries across different metrics

## Installation and Running

1. Ensure you have Python installed (Python 3.7 or higher recommended)
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
<!-- 3. Generate the dataset (if needed):
   ```
   python create_dataset.py
   ``` -->
4. Run the Shiny app:
   ```
   shiny run app.py
   ```
5. Open your browser and navigate to the URL shown in the terminal (usually http://127.0.0.1:8000)

## Data Sources

This application uses data from:
- World Happiness Report
- Life Expectancy and Economic Indicators dataset
- World Development Indicators

## Customization

Each tab has a customized sidebar with specific controls relevant to that visualization. You can adjust various parameters to explore different aspects of the data.

## Purpose



Your dashboard should have at least 5 charts, and you should use at least 3 types of charts.

---

Title: Understanding Global Well-Being: A Dashboard Exploration of Happiness and Life Expectancy

In this project, we aim to explore the relationship between happiness, life expectancy, and socio-economic indicators across countries using two complementary datasets: the World Happiness Report and the World Bank Life Expectancy and Socioeconomic Indicators dataset.

The central question we address is: “How do socio-economic factors and perceived happiness relate to life expectancy across different regions of the world?”

This question is significant because it connects subjective well-being (happiness) with objective measures of quality of life (life expectancy, GDP, education, etc.), offering insights that can inform public health policy, global development goals, and well-being optimization strategies.

The challenge lies in the heterogeneity and granularity of the data. The happiness dataset focuses on perception-based survey results, while the life expectancy dataset includes time series and numerical indicators. Combining and visualizing these datasets in a cohesive story that is both intuitive and interactive is non-trivial. Moreover, capturing multivariate relationships visually—across regions, time, and development levels—demands thoughtful dashboard design and chart selection.

Our dashboard leverages Python Shiny for interactivity, allowing users to explore trends, filter by region, and observe correlations dynamically. The story unfolds in multiple parts: global overviews, regional breakdowns, correlation insights, and trend analysis, all within a clean and logical layout.