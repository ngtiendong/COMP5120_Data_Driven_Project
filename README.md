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