from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load the dataset
data = pd.read_csv('data/all_with_happiness.csv')

# Generate factor choices dynamically from numeric columns
# Exclude identifier and categorical columns
exclude_columns = ['Country', 'Year', 'Code', 'Year_Code', 'Life Expectancy', 'Life Expectancy Male', 'Life Expectancy Female', 'Happiness Score']
factor_choices = [col for col in data.columns.tolist() if col not in exclude_columns]

# Load continents data
with open('data/config/continents.json', 'r') as f:
    continents_data = json.load(f)
continent_choices = list(continents_data.keys())

# Convert country names from NumPy array to list
country_choices = list(data['Country'].unique())
year_choices = list(data['Year'].unique()) if 'Year' in data.columns else [2019]

# Define UI with tabs
app_ui = ui.page_fluid(
    ui.include_css("www/styles.css"),
    ui.panel_title("Life Expectancy, Happiness, and Economic Analysis"),
    
    # Horizontal tabs on top
    ui.navset_tab(
        # Tab 1 - World Map Overview
        ui.nav_panel("World Map Overview", 
            ui.row(
                ui.column(3,
                    ui.h4("Map Controls"),
                    ui.input_slider("year_range_map", "Select Year Range", 
                                  min=min(year_choices), max=max(year_choices), 
                                  value=[min(year_choices), max(year_choices)], step=1),
                    ui.input_checkbox("show_average", "Show Average Over Range", value=True),
                    ui.input_select("color_scale", "Color Scale", 
                                  choices=["Viridis", "Plasma", "Blues", "Reds", "Greens"],
                                  selected="Viridis"),
                    ui.hr(),
                    ui.h4("Country Details"),
                    ui.input_select("country1", "Select a Country", choices=country_choices, selected="Vietnam"),
                    ui.output_ui("country_details"),
                    ui.hr(),
                    ui.h4("Select Factor for Color Intensity"),
                    ui.input_radio_buttons("factor_radio", "Select Factor", 
                                         choices=["Life Expectancy", "Happiness Score", "GDP per Capita"],
                                         selected="Life Expectancy",
                                         inline=True)
                ),
                ui.column(9,
                    ui.output_ui("world_map_plot")
                )
            )
        ),
        
        # Tab 2 - Happiness, GDP, and Life Expectancy Trends
        ui.nav_panel("Happiness, GDP, and Life Expectancy Trends", 
            ui.row(
                ui.column(3,
                    ui.h4("Controls"),
                    ui.input_select("view_type", "View Type", 
                                  choices=["Country View", "Continent View"],
                                  selected="Country View"),
                    ui.output_ui("dynamic_selector"),
                    ui.input_slider("years_range2", "Year Range", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=[min(year_choices), max(year_choices)], step=1)
                ),
                ui.column(9,
                    ui.h3("Happiness, GDP, and Life Expectancy Trends"),
                    ui.output_ui("happiness_plot")
                )
            )
        ),

        # Tab 3 - Factors Indicator
        ui.nav_panel("Factors Indicator", 
            ui.row(
                ui.column(3,
                    ui.h4("Controls"),
                    ui.input_select("view_type3", "View Type", 
                                  choices=["Country View", "Continent View"],
                                  selected="Country View"),
                    ui.output_ui("dynamic_selector3"),
                    ui.input_slider("years_range3", "Year Range", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=[min(year_choices), max(year_choices)], step=1),
                    ui.hr(),
                    ui.input_selectize("factors3", "Select Factors to Visualize", 
                                     choices=factor_choices,
                                     multiple=True,
                                     selected=["GDP per Capita", "Government Health Expenditure", "CO2 Emissions Total", "Happiness Score"]),
                    ui.hr(),
                ),
                ui.column(9,
                    ui.h3("Time Series Analysis: Factors Affecting Life Expectancy"),
                    ui.output_ui("economic_plot"),
                    ui.hr(),
                    ui.h4("Spider Chart: Factor Impact on Life Expectancy"),
                    ui.row(
                        ui.column(12, ui.output_ui("spider_total")),
                    )
                )
            )
        ),
        
        # Tab 4 - Comparative Analysis
        ui.nav_panel("Factors Indicator by Top Countries", 
            ui.row(
                ui.column(3,
                    ui.h4("Country Selection Controls"),
                    ui.input_select("rank_type", "Select Countries By", 
                                  choices=["Highest Life Expectancy", "Lowest Life Expectancy"],
                                  selected="Highest Life Expectancy"),
                    ui.input_select("num_countries", "Number of Countries", 
                                  choices=["5", "10", "15", "20"],
                                  selected="10"),
                    ui.input_slider("years_range4", "Year Range for Ranking", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=[min(year_choices), max(year_choices)], step=1),
                    ui.hr(),
                    ui.input_selectize("factors4", "Select Factors to Visualize", 
                                     choices=factor_choices,
                                     multiple=True,
                                     selected=["GDP per Capita", "GDP", "Government Health Expenditure", "CO2 Emissions Total", "Happiness Score"]),
                    ui.hr(),
                    ui.output_ui("selected_countries_list")
                ),
                ui.column(9,
                    ui.output_ui("comparison_correlation_plot")
                )
            )
        ),
        
        # Tab 5 - Predictive Analysis
        ui.nav_panel("Predictive Analysis using ML", 
            ui.row(
                ui.column(3,
                    ui.h4("Regression Controls"),
                    ui.input_selectize("regression_factors", "Select Factors for Regression", 
                                     choices=factor_choices,
                                     multiple=True,
                                     selected=["GDP per Capita", "Government Health Expenditure", "GDP", "Total Health Expenditure", "CO2 Emissions Total", "CO2 Emissions Agriculture"]),
                    ui.input_action_button("run_regression", "Run Regression Analysis"),
                    ui.hr(),
                    ui.h4("Regression Results"),
                    ui.output_ui("regression_results")
                ),
                ui.column(9,
                    ui.h3("Predictive Analysis: Factors Affecting Life Expectancy & Happiness"),
                    ui.h4("Life Expectancy Prediction"),
                    ui.output_ui("life_expectancy_plot"),
                    ui.hr(),
                    ui.h4("Happiness Score Prediction"),
                    ui.output_ui("happiness_prediction_plot"),
                    ui.hr(),
                    ui.h4("Feature Importance Comparison"),
                    ui.output_ui("feature_importance"),
                    ui.hr(),
                    ui.h4("Correlation Analysis"),
                    ui.output_ui("correlation_results")
                )
            )
        )
    )
)

def server(input, output, session):
    # Tab 1 - World Map of Life Expectancy
    @output
    @render.ui
    def world_map_plot():
        # Get slider values (these should already be initialized to min/max of year_choices)
        year_min, year_max = input.year_range_map()
        year_min = int(year_min)
        year_max = int(year_max)
        show_average = input.show_average()
        selected_factor = input.factor_radio()

        # Filter data for the selected year range
        year_data = data[(data['Year'] >= year_min) & (data['Year'] <= year_max)]
        
        if year_data.empty:
            fig = px.choropleth(
                title="No data available for the selected year range"
            )
            fig.add_annotation(text="Please adjust your year range", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # If show_average is checked (default is True), calculate and show averages
        if show_average:
            try:
                # Calculate average life expectancy for each country in the selected range
                # Using more robust aggregation
                numeric_cols = ['Life Expectancy', 'Happiness Score', 'GDP per Capita']
                # categorical_cols = ['Region', 'Income Group']
                
                # First handle numeric columns with mean of values > 0
                def mean_positive(series):
                    positive_values = series[series > 0]
                    return positive_values.mean() if len(positive_values) > 0 else 0
                
                avg_numeric = year_data.groupby('Country')[numeric_cols].agg(mean_positive)
                
                # Then handle categorical columns by taking first non-null value
                # Use a lambda function to prevent string concatenation issues
                # avg_categorical = year_data.groupby('Country').agg({
                #     col: lambda x: x.iloc[0] if len(x) > 0 else None 
                #     for col in categorical_cols
                # })
                
                # Combine the results
                avg_data = pd.concat([avg_numeric], axis=1).reset_index()
                
                # Create title with range information
                title = f"Average Info (Life Expectancy, GDP, Happiness Score) by Country ({int(year_min)}-{int(year_max)})"
                plot_data = avg_data
            except Exception as e:
                # Fallback to using the most recent year if aggregation fails
                print(f"Error in aggregation: {e}")  # For debugging
                title = f"Life Expectancy, GDP, Happiness Score by Country ({int(year_max)}) - Showing latest year"
                plot_data = year_data[year_data['Year'] == year_max]
        else:
            # Use the most recent year in the range
            title = f"Life Expectancy, GDP, Happiness Score by Country ({int(year_max)})"
            plot_data = year_data[year_data['Year'] == year_max]
            
            # If no data for the most recent year, use the last available year
            if plot_data.empty and not year_data.empty:
                last_available_year = year_data['Year'].max()
                plot_data = year_data[year_data['Year'] == last_available_year]
                title = f"Life Expectancy, GDP, Happiness Score by Country ({int(last_available_year)}) - Most recent available data"

        # Create a choropleth map
        fig = px.choropleth(
            plot_data,
            locations="Country",
            locationmode="country names",
            color=selected_factor,
            hover_name="Country",
            color_continuous_scale=input.color_scale().lower(),
            title=title,
            hover_data={
                "Life Expectancy": ":.1f",
                "Happiness Score": ":.2f", 
                "GDP per Capita": ":,.0f"
            } if all(col in plot_data.columns for col in ["Life Expectancy", "Happiness Score", "GDP per Capita"]) else None,
            projection="natural earth"
        )
        
        # Improve layout
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            margin={"r":0,"t":50,"l":0,"b":0},
            height=600
        )

        # Convert to Shiny UI
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    # Tab 1 - Country Details
    @output
    @render.ui
    def country_details():
        selected_country = input.country1()
        year_min, year_max = input.year_range_map()
        show_average = input.show_average()
        
        try:
            # Filter for selected country and year range
            country_data = data[(data['Country'] == selected_country) & 
                               (data['Year'] >= year_min) & 
                               (data['Year'] <= year_max)]
            
            if country_data.empty:
                return ui.div(
                    ui.p(f"No data available for {selected_country} in selected years", 
                         style="color: #666; font-style: italic;")
                )
            
            if show_average:
                # Get only numeric columns we need for averaging
                columns_to_average = ['Life Expectancy', 'Happiness Score', 'GDP per Capita']
                
                # Calculate averages excluding zero values
                def mean_positive(series):
                    positive_values = series[series > 0]
                    return positive_values.mean() if len(positive_values) > 0 else 0
                
                life_exp = mean_positive(country_data['Life Expectancy'])
                happiness = mean_positive(country_data['Happiness Score'])
                gdp = mean_positive(country_data['GDP per Capita'])
                
                return ui.div(
                    ui.p("📍 Country:", ui.strong(f" {selected_country}"), style="margin-bottom: 8px;"),
                    ui.p("📅 Time Period:", ui.strong(f" {year_min}-{year_max} (Average)"), style="margin-bottom: 8px;"),
                    ui.p("💗 Life Expectancy:", ui.strong(f" {life_exp:.1f} years"), style="margin-bottom: 8px;"),
                    ui.p("😊 Happiness Score:", ui.strong(f" {happiness:.2f}/10"), style="margin-bottom: 8px;"),
                    ui.p("💰 GDP per Capita (Million):", ui.strong(f" ${gdp:,.0f}"), style="margin-bottom: 8px;"),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;"
                )
            else:
                # Show most recent year
                recent_data = country_data[country_data['Year'] == year_max]
                
                if recent_data.empty:
                    return ui.div(
                        ui.p(f"No data available for {selected_country} in {year_max}", 
                             style="color: #666; font-style: italic;")
                    )
                
                life_exp = recent_data['Life Expectancy'].values[0]
                happiness = recent_data['Happiness Score'].values[0]
                gdp = recent_data['GDP per Capita'].values[0]
                
                return ui.div(
                    ui.p("📍 Country:", ui.strong(f" {selected_country}"), style="margin-bottom: 8px;"),
                    ui.p("📅 Year:", ui.strong(f" {year_max}"), style="margin-bottom: 8px;"),
                    ui.p("💗 Life Expectancy:", ui.strong(f" {life_exp:.1f} years"), style="margin-bottom: 8px;"),
                    ui.p("😊 Happiness Score:", ui.strong(f" {happiness:.2f}/10"), style="margin-bottom: 8px;"),
                    ui.p("💰 GDP per Capita (Million):", ui.strong(f" ${gdp:,.0f}"), style="margin-bottom: 8px;"),
                    style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;"
                )
        except (IndexError, KeyError):
            return ui.div(
                ui.p(f"No data available for {selected_country}", 
                     style="color: #666; font-style: italic;")
            )
        
    # Tab 2 - Dynamic selector for country/continent
    @output
    @render.ui
    def dynamic_selector():
        view_type = input.view_type()
        
        if view_type == "Country View":
            return ui.input_select("selected_location", "Select a Country", 
                                 choices=country_choices, selected="Vietnam")
        else:
            return ui.input_select("selected_location", "Select a Continent", 
                                 choices=continent_choices, selected="Asia")
    
    # Tab 2 - Happiness Plot  
    @output
    @render.ui
    def happiness_plot():
        view_type = input.view_type()
        year_min, year_max = input.years_range2()
        
        # Get the selected location from the dynamic selector
        try:
            selected_location = input.selected_location()
        except:
            # If the input doesn't exist yet, return a loading message
            fig = px.scatter(title="Loading...")
            fig.add_annotation(text="Please select a location", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if view_type == "Country View":
            # Filter data for the selected country and years
            filtered_data = data[(data['Country'] == selected_location) & 
                               (data['Year'] >= year_min) & 
                               (data['Year'] <= year_max)]
            
            if filtered_data.empty:
                fig = px.scatter(title=f"No data available for {selected_location}")
                fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
            
            # Check if required columns exist
            required_columns = ['Life Expectancy', 'Happiness Score', 'GDP per Capita']
            missing_columns = [col for col in required_columns if col not in filtered_data.columns]
            
            if missing_columns:
                fig = px.scatter(title=f"Missing data columns for {selected_location}")
                fig.add_annotation(text=f"Missing: {', '.join(missing_columns)}", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
            
            plot_data = filtered_data
            title_location = selected_location

        else:
            # Continent view - calculate average for countries in the continent
            if selected_location in continents_data:
                continent_countries = continents_data[selected_location]
                
                # Filter for countries in the continent and the selected years
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                
                if continent_data.empty:
                    fig = px.scatter(title=f"No data available for countries in {selected_location}")
                    fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                # Check if required columns exist
                required_columns = ['Life Expectancy', 'Happiness Score', 'GDP per Capita']
                missing_columns = [col for col in required_columns if col not in continent_data.columns]
                
                if missing_columns:
                    fig = px.scatter(title=f"Missing data columns for {selected_location}")
                    fig.add_annotation(text=f"Missing: {', '.join(missing_columns)}", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                # Calculate average for the continent - only include columns that exist
                available_factor_cols = [col for col in factor_choices if col in continent_data.columns]
                agg_dict = {col: 'mean' for col in available_factor_cols}
                
                # Ensure we always include the required columns in aggregation
                for col in required_columns:
                    if col in continent_data.columns:
                        agg_dict[col] = 'mean'
                
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                
                # Check if aggregated data has required columns
                if not all(col in plot_data.columns for col in required_columns):
                    fig = px.scatter(title=f"Insufficient data for {selected_location}")
                    fig.add_annotation(text="Missing required columns after aggregation", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                title_location = selected_location
            else:
                fig = px.scatter(title=f"No data available for {selected_location}")
                fig.add_annotation(text="Invalid continent selection", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Create three separate plots without normalization
        plot_data_viz = plot_data.copy()
        
        # Replace 0 values with NaN for better visualization
        plot_data_viz['Happiness Score'] = plot_data_viz['Happiness Score'].replace(0, np.nan)
        plot_data_viz['GDP per Capita'] = plot_data_viz['GDP per Capita'].replace(0, np.nan)
        plot_data_viz['Life Expectancy'] = plot_data_viz['Life Expectancy'].replace(0, np.nan)
        
        # Create subplot figure with 3 plots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f"Life Expectancy for {title_location} ({year_min}-{year_max})",
                f"GDP per Capita for {title_location} ({year_min}-{year_max})",
                f"Happiness Score for {title_location} ({year_min}-{year_max})"
            ),
            vertical_spacing=0.1
        )
        
        # Add Life Expectancy plot
        fig.add_scatter(
            x=plot_data_viz['Year'], 
            y=plot_data_viz['Life Expectancy'],
            mode='lines+markers', 
            name='Life Expectancy',
            line=dict(color='red', width=3),
            row=1, col=1
        )
        
        # Add GDP per Capita plot
        fig.add_scatter(
            x=plot_data_viz['Year'], 
            y=plot_data_viz['GDP per Capita'],
            mode='lines+markers', 
            name='GDP per Capita',
            line=dict(color='green', width=3),
            row=2, col=1
        )
        
        # Add Happiness Score plot
        fig.add_scatter(
            x=plot_data_viz['Year'], 
            y=plot_data_viz['Happiness Score'],
            mode='lines+markers', 
            name='Happiness Score',
            line=dict(color='blue', width=3),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Years", row=1, col=1)
        fig.update_yaxes(title_text="USD (Millions)", row=2, col=1)
        fig.update_yaxes(title_text="Score (0-10)", row=3, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Year", row=3, col=1)
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
    # Tab 3 - Dynamic selector for country/continent
    @output
    @render.ui
    def dynamic_selector3():
        view_type = input.view_type3()
        
        if view_type == "Country View":
            return ui.input_select("selected_location3", "Select a Country", 
                                 choices=country_choices, selected="Vietnam")
        else:
            return ui.input_select("selected_location3", "Select a Continent", 
                                 choices=continent_choices, selected="Asia")

    # Tab 3 - Economic Plot
    @output
    @render.ui
    def economic_plot():
        view_type = input.view_type3()
        year_min, year_max = input.years_range3()
        selected_factors = input.factors3()
        
        if not selected_factors:
            # Return message if no factors selected
            fig = px.scatter(title="Please select at least one factor to visualize")
            fig.add_annotation(text="Select factors from the control panel", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get the selected location from the dynamic selector
        try:
            selected_location = input.selected_location3()
        except:
            # If the input doesn't exist yet, return a loading message
            fig = px.scatter(title="Loading...")
            fig.add_annotation(text="Please select a location", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if view_type == "Country View":
            # Filter data for the selected country and years
            filtered_data = data[(data['Country'] == selected_location) & 
                               (data['Year'] >= year_min) & 
                               (data['Year'] <= year_max)]
            
            if filtered_data.empty:
                fig = px.scatter(title=f"No data available for {selected_location}")
                fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
            
            title_location = selected_location
            plot_data = filtered_data
            
        else:  # Continent View
            if selected_location in continents_data:
                continent_countries = continents_data[selected_location]
                
                # Filter for countries in the continent and the selected years
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                
                if continent_data.empty:
                    fig = px.scatter(title=f"No data available for countries in {selected_location}")
                    fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                # Calculate average by year for the continent - include Life Expectancy
                agg_dict = {col: 'mean' for col in factor_choices + ['Life Expectancy'] if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                
                title_location = selected_location
            else:
                fig = px.scatter(title=f"No data available for {selected_location}")
                fig.add_annotation(text="Invalid continent selection", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Replace 0 values with NaN for better visualization
        plot_data_viz = plot_data.copy()
        for factor in list(selected_factors) + ['Life Expectancy']:
            if factor in plot_data_viz.columns:
                plot_data_viz[factor] = plot_data_viz[factor].replace(0, np.nan)
        
        # Create time series line plot
        fig = px.line()
        
        # Add line for Life Expectancy (always shown as reference)
        fig.add_scatter(x=plot_data_viz['Year'], y=plot_data_viz['Life Expectancy'], 
                      mode='lines+markers', name='Life Expectancy (Reference)', 
                      line=dict(color='red', width=3))
        
        # Color palette for factors
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Add lines for selected factors
        for i, factor in enumerate(selected_factors):
            if factor in plot_data_viz.columns:
                # Get non-NaN values for scaling calculations
                valid_factor_values = plot_data_viz[factor].dropna()
                valid_life_exp_values = plot_data_viz['Life Expectancy'].dropna()
                
                if len(valid_factor_values) > 0 and len(valid_life_exp_values) > 0:
                    # Normalize factor to same scale as life expectancy for better comparison
                    if factor == 'Happiness Score':
                        # Scale happiness score (0-10) to life expectancy range
                        normalized_values = plot_data_viz[factor] * (valid_life_exp_values.max() / 10)
                    elif factor == 'GDP per Capita':
                        # Scale GDP to life expectancy range
                        max_factor = valid_factor_values.max()
                        if max_factor > 0:
                            normalized_values = (plot_data_viz[factor] / max_factor) * valid_life_exp_values.max()
                        else:
                            normalized_values = plot_data_viz[factor]  # Keep original if max is 0
                    else:
                        # For other factors, scale to life expectancy range
                        max_factor = valid_factor_values.max()
                        if max_factor > 0:
                            normalized_values = (plot_data_viz[factor] / max_factor) * valid_life_exp_values.max()
                        else:
                            normalized_values = plot_data_viz[factor]  # Keep original if max is 0
                    
                    line_style = dict(color=colors[i % len(colors)], width=2)
                    
                    fig.add_scatter(x=plot_data_viz['Year'], y=normalized_values, 
                                  mode='lines+markers', name=factor, 
                                  line=line_style)
        
        # Update layout
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Normalized Values (Life Expectancy Scale)",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=600,
            width=900
        )
        
        return ui.div(
            ui.HTML(fig.to_html(include_plotlyjs="cdn")),
            ui.div(
                ui.p(f"Time Series Analysis for {title_location} ({year_min}-{year_max})",
                     style="text-align: center; font-size: 14px; color: #666; margin-top: 10px; font-style: italic;")
            )
        )
        
    # Tab 4 - Get selected countries based on ranking
    @reactive.Calc
    def get_ranked_countries():
        rank_type = input.rank_type()
        num_countries = int(input.num_countries())
        year_min, year_max = input.years_range4()
        
        # Filter data for the selected year range
        year_data = data[(data['Year'] >= year_min) & (data['Year'] <= year_max)]
        
        if year_data.empty:
            return []
        
        # Calculate average life expectancy for each country in the range
        avg_life_exp = year_data.groupby('Country')['Life Expectancy'].mean().reset_index()
        
        # Sort based on rank type
        if rank_type == "Highest Life Expectancy":
            ranked_countries = avg_life_exp.nlargest(num_countries, 'Life Expectancy')
        else:
            ranked_countries = avg_life_exp.nsmallest(num_countries, 'Life Expectancy')
        
        return ranked_countries['Country'].tolist()
    
    # Tab 4 - Show selected countries list
    @output
    @render.ui
    def selected_countries_list():
        countries = get_ranked_countries()
        rank_type = input.rank_type()
        num_countries = input.num_countries()
        
        if not countries:
            return ui.div(
                ui.p("No countries available for selection", 
                     style="color: #666; font-style: italic;")
            )
        
        title = f"Top {num_countries} Countries" if rank_type == "Highest Life Expectancy" else f"Bottom {num_countries} Countries"
        
        country_list = ui.div(
            ui.h5(title),
            *[ui.p(f"{i+1}. {country}", style="margin-bottom: 4px;") for i, country in enumerate(countries)],
            style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;"
        )
        
        return country_list
    
    # Tab 4 - Correlation/Pair Plot Analysis
    @output
    @render.ui
    def comparison_correlation_plot():
        countries = get_ranked_countries()
        selected_factors = input.factors4()
        year_min, year_max = input.years_range4()
        rank_type = input.rank_type()
        num_countries = input.num_countries()
        
        if not countries:
            fig = px.scatter(title="No countries selected")
            fig.add_annotation(text="Adjust your year range", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if not selected_factors:
            fig = px.scatter(title="Please select at least one factor to visualize")
            fig.add_annotation(text="Select factors from the control panel", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Filter data for selected countries and years
        filtered_data = data[(data['Country'].isin(countries)) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
        
        if filtered_data.empty:
            fig = px.scatter(title="No data available for selected countries")
            fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Prepare data for correlation analysis
        correlation_cols = ['Life Expectancy'] + list(selected_factors)
        corr_data = filtered_data[correlation_cols + ['Country']].copy()
        
        # Remove 0 values (which are placeholders for missing data) before analysis
        for col in correlation_cols:
            if col in corr_data.columns:
                corr_data[col] = corr_data[col].replace(0, np.nan)
        
        # Drop rows with any NaN values after replacing 0s
        corr_data = corr_data.dropna(subset=correlation_cols)
        
        if corr_data.empty or len(corr_data) < 2:
            fig = px.scatter(title="Insufficient data for correlation analysis")
            fig.add_annotation(text="Need more data points (after removing 0 values)", showarrow=False, font=dict(size=20))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        title_text = f"{rank_type.replace('Life Expectancy', '').strip()} {num_countries} Countries"
        
        try:
            # Set seaborn style
            plt.style.use('default')
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            
            # Create correlation matrix using seaborn (now with 0 values removed)
            correlation_matrix = corr_data[correlation_cols].corr()
            
            # Create correlation heatmap using seaborn
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(
                correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                fmt='.2f',
                annot_kws={'size': 10}
            )
            
            ax_corr.set_title(f'Correlation Matrix: {title_text} ({year_min}-{year_max})\n(Zero values excluded)', 
                             fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Convert to base64 for HTML
            buffer_corr = BytesIO()
            fig_corr.savefig(buffer_corr, format='png', dpi=150, bbox_inches='tight')
            buffer_corr.seek(0)
            img_corr_b64 = base64.b64encode(buffer_corr.getvalue()).decode()
            plt.close(fig_corr)
            
            # Create pair plot using seaborn if we have multiple factors
            if len(selected_factors) > 1:
                # Limit to most important factors for better visualization
                plot_factors = ['Life Expectancy'] + list(selected_factors)[:4]  # Max 5 factors
                pair_data = corr_data[plot_factors + ['Country']]
                
                # Create pair plot
                fig_pair = plt.figure(figsize=(12, 10))
                
                # Use seaborn pairplot (data already has 0 values removed)
                g = sns.pairplot(
                    pair_data[plot_factors], 
                    diag_kind='hist',
                    plot_kws={'alpha': 0.6, 's': 30},
                    diag_kws={'bins': 20, 'alpha': 0.7}
                )
                
                g.fig.suptitle(f'Pair Plot: {title_text} ({year_min}-{year_max})\n(Zero values excluded)', 
                              fontsize=14, fontweight='bold', y=1.02)
                
                # Add correlation coefficients to upper triangle
                for i in range(len(plot_factors)):
                    for j in range(i+1, len(plot_factors)):
                        ax = g.axes[i, j]
                        corr_val = correlation_matrix.loc[plot_factors[i], plot_factors[j]]
                        ax.text(0.5, 0.5, f'r = {corr_val:.2f}', 
                               transform=ax.transAxes, 
                               ha='center', va='center',
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor='white', 
                                        edgecolor='gray',
                                        alpha=0.8))
                        ax.set_xlabel('')
                        ax.set_ylabel('')
                
                plt.tight_layout()
                
                # Convert to base64
                buffer_pair = BytesIO()
                g.fig.savefig(buffer_pair, format='png', dpi=150, bbox_inches='tight')
                buffer_pair.seek(0)
                img_pair_b64 = base64.b64encode(buffer_pair.getvalue()).decode()
                plt.close(g.fig)
                
                pair_html = f'<img src="data:image/png;base64,{img_pair_b64}" style="width:100%; max-width:900px; height:auto;">'
                
            else:
                # Single factor - create simple scatter plot with regression line
                factor = selected_factors[0]
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                
                # Create scatter plot with regression line (data already has 0 values removed)
                sns.regplot(
                    data=corr_data, 
                    x=factor, 
                    y='Life Expectancy',
                    scatter_kws={'alpha': 0.6, 's': 50},
                    line_kws={'color': 'red', 'linewidth': 2}
                )
                
                # Calculate and display correlation
                corr_val = corr_data[factor].corr(corr_data['Life Expectancy'])
                data_points = len(corr_data)
                ax_scatter.text(0.05, 0.95, f'Correlation: r = {corr_val:.3f}\nData points: {data_points}', 
                               transform=ax_scatter.transAxes,
                               fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.5", 
                                        facecolor='yellow', 
                                        alpha=0.8))
                
                ax_scatter.set_title(f'Life Expectancy vs {factor}: {title_text} ({year_min}-{year_max})\n(Zero values excluded)', 
                                   fontsize=14, fontweight='bold')
                ax_scatter.set_xlabel(factor, fontsize=12)
                ax_scatter.set_ylabel('Life Expectancy (years)', fontsize=12)
                ax_scatter.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Convert to base64
                buffer_scatter = BytesIO()
                fig_scatter.savefig(buffer_scatter, format='png', dpi=150, bbox_inches='tight')
                buffer_scatter.seek(0)
                img_scatter_b64 = base64.b64encode(buffer_scatter.getvalue()).decode()
                plt.close(fig_scatter)
                
                pair_html = f'<img src="data:image/png;base64,{img_scatter_b64}" style="width:100%; max-width:800px; height:auto;">'
            
            # Create distribution plots for key factors (data already has 0 values removed)
            fig_dist, axes_dist = plt.subplots(2, 2, figsize=(12, 8))
            axes_dist = axes_dist.flatten()
            
            # Plot distributions for up to 4 factors
            dist_factors = ['Life Expectancy'] + list(selected_factors)[:3]
            
            for i, factor in enumerate(dist_factors):
                if i < 4:
                    sns.histplot(
                        data=corr_data, 
                        x=factor, 
                        kde=True, 
                        ax=axes_dist[i],
                        alpha=0.7,
                        color=sns.color_palette("husl", len(dist_factors))[i]
                    )
                    axes_dist[i].set_title(f'Distribution of {factor}', fontweight='bold')
                    axes_dist[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(dist_factors), 4):
                axes_dist[i].set_visible(False)
            
            data_points_total = len(corr_data)
            fig_dist.suptitle(f'Factor Distributions: {title_text}\n(Zero values excluded, n={data_points_total})', 
                             fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Convert to base64
            buffer_dist = BytesIO()
            fig_dist.savefig(buffer_dist, format='png', dpi=150, bbox_inches='tight')
            buffer_dist.seek(0)
            img_dist_b64 = base64.b64encode(buffer_dist.getvalue()).decode()
            plt.close(fig_dist)
            
            # Combine all plots in HTML
            combined_html = f"""
            <div style="text-align: center;">
                <h3>Correlation Analysis: {title_text} ({year_min}-{year_max})</h3>
                
                <div style="margin-bottom: 30px;">
                    <h4>📊 Correlation Heatmap</h4>
                    <img src="data:image/png;base64,{img_corr_b64}" style="width:100%; max-width:800px; height:auto;">
                </div>
                
                <div style="margin-bottom: 30px;">
                    <h4>🔗 Factor Relationships</h4>
                    {pair_html}
                </div>
                
                <div style="margin-bottom: 20px;">
                    <h4>📈 Factor Distributions</h4>
                    <img src="data:image/png;base64,{img_dist_b64}" style="width:100%; max-width:900px; height:auto;">
                </div>
                
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: left;">
                    <h5>📋 Analysis Summary:</h5>
                    <ul>
                        <li><strong>Data Quality:</strong> Zero values (missing data placeholders) have been excluded from analysis</li>
                        <li><strong>Sample Size:</strong> {data_points_total} valid data points after cleaning</li>
                        <li><strong>Strong Correlations (|r| > 0.7):</strong> Highly related factors</li>
                        <li><strong>Moderate Correlations (0.3 < |r| < 0.7):</strong> Some relationship</li>
                        <li><strong>Weak Correlations (|r| < 0.3):</strong> Little to no linear relationship</li>
                        <li><strong>Positive values:</strong> Factors increase together</li>
                        <li><strong>Negative values:</strong> One factor increases as other decreases</li>
                    </ul>
                </div>
            </div>
            """
            
            return ui.HTML(combined_html)
            
        except Exception as e:
            # Fallback to simple correlation display
            fig = px.scatter(title=f"Correlation Analysis: {title_text}")
            fig.add_annotation(text=f"Error creating seaborn plots: {str(e)[:100]}...", 
                              showarrow=False, font=dict(size=14))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    # Tab 4 - Spider Charts (REMOVED)
    # @output
    # @render.ui
    # def comparison_spider_total():
    #     # This function has been removed to simplify Tab 4
    #     pass
    
    # Tab 5 - Enhanced Regression Model with multiple algorithms for both targets
    @reactive.Calc
    def regression_model():
        # This will be triggered when the button is clicked
        input.run_regression()  # This makes the calculation reactive to button clicks
        
        selected_factors = input.regression_factors()
        
        if not selected_factors:
            return None
        
        # Convert tuple to list if necessary
        selected_factors = list(selected_factors)
        
        # Prepare data for regression - analyze both targets
        target_variables = ['Life Expectancy', 'Happiness Score']
        required_columns = target_variables + selected_factors
        reg_data = data.dropna(subset=required_columns)
        
        # Remove 0 values from target variables and factors
        for col in required_columns:
            if col in reg_data.columns:
                reg_data = reg_data[reg_data[col] > 0]
        
        if len(reg_data) < 10:
            return None
        
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            X = reg_data[selected_factors]
            
            # Analyze both targets
            results = {}
            
            for target in target_variables:
                y = reg_data[target]
                
                # Split data for proper evaluation
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features for Linear Regression
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Define models to test
                models = {
                    'Linear Regression': {
                        'model': LinearRegression(),
                        'use_scaling': True
                    },
                    'Random Forest': {
                        'model': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                        'use_scaling': False
                    },
                    'Gradient Boosting': {
                        'model': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
                        'use_scaling': False
                    }
                }
                
                model_results = {}
                
                # Train and evaluate each model
                for name, model_config in models.items():
                    model = model_config['model']
                    use_scaling = model_config['use_scaling']
                    
                    # Choose appropriate training data
                    if use_scaling:
                        X_train_final = X_train_scaled
                        X_test_final = X_test_scaled
                    else:
                        X_train_final = X_train
                        X_test_final = X_test
                    
                    # Train the model
                    model.fit(X_train_final, y_train)
                    
                    # Make predictions
                    y_pred_train = model.predict(X_train_final)
                    y_pred_test = model.predict(X_test_final)
                    
                    # Calculate metrics
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    
                    # Store results
                    model_results[name] = {
                        'model': model,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'rmse': test_rmse,
                        'mae': test_mae,
                        'use_scaling': use_scaling,
                        'y_pred_test': y_pred_test,
                        'overfitting': train_r2 - test_r2
                    }
                
                # Select best model based on test R² score
                best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
                best_model_info = model_results[best_model_name]
                best_model = best_model_info['model']
                
                # Make predictions on full dataset using the best model
                if best_model_info['use_scaling']:
                    X_full_scaled = scaler.fit_transform(X)
                    y_pred_full = best_model.fit(X_full_scaled, y).predict(X_full_scaled)
                else:
                    y_pred_full = best_model.fit(X, y).predict(X)
                
                # Calculate final metrics on full dataset
                final_r2 = r2_score(y, y_pred_full)
                final_rmse = np.sqrt(mean_squared_error(y, y_pred_full))
                
                # Create results summary for the best model
                if hasattr(best_model, 'coef_'):
                    coefficients = best_model.coef_
                elif hasattr(best_model, 'feature_importances_'):
                    coefficients = best_model.feature_importances_
                else:
                    coefficients = np.ones(len(selected_factors))
                
                results_list = []
                for i, factor in enumerate(selected_factors):
                    coef_value = coefficients[i]
                    results_list.append({
                        'Factor': factor,
                        'Coefficient': coef_value,
                        'Impact': 'Positive' if coef_value > 0 else 'Negative'
                    })
                
                results_df = pd.DataFrame(results_list)
                
                # Feature importance based on absolute coefficients/importances
                importance_df = pd.DataFrame({
                    'Factor': selected_factors,
                    'Importance': abs(coefficients),
                    'Coefficient': coefficients,
                    'Impact': ['Positive' if coef > 0 else 'Negative' for coef in coefficients]
                }).sort_values('Importance', ascending=True)
                
                # Model comparison summary
                comparison_df = pd.DataFrame({
                    'Model': list(model_results.keys()),
                    'Test R²': [model_results[name]['test_r2'] for name in model_results.keys()],
                    'RMSE': [model_results[name]['rmse'] for name in model_results.keys()],
                    'MAE': [model_results[name]['mae'] for name in model_results.keys()],
                    'Overfitting': [model_results[name]['overfitting'] for name in model_results.keys()]
                }).sort_values('Test R²', ascending=False)
                
                # Store results for this target
                results[target] = {
                    'best_model': best_model,
                    'best_model_name': best_model_name,
                    'data': reg_data,
                    'X': X,
                    'y': y,
                    'y_pred': y_pred_full,
                    'r2': final_r2,
                    'rmse': final_rmse,
                    'results_df': results_df,
                    'importance_df': importance_df,
                    'selected_factors': selected_factors,
                    'model_comparison': comparison_df,
                    'all_models': model_results,
                    'scaler': scaler if best_model_info['use_scaling'] else None
                }
            
            return {
                'life_expectancy': results['Life Expectancy'],
                'happiness_score': results['Happiness Score'],
                'data': reg_data
            }
            
        except ImportError:
            return {'error': 'ImportError', 'message': 'scikit-learn not available. Install with: pip install scikit-learn'}
        except Exception as e:
            return {'error': 'Exception', 'message': f'Error in regression analysis: {str(e)}'}

    @output
    @render.ui
    def regression_results():
        model_result = regression_model()
        
        if model_result is None:
            return ui.div(
                ui.p("Click 'Run Regression Analysis' to see results", 
                     style="color: #666; font-style: italic;")
            )
        
        if 'error' in model_result:
            return ui.div(
                ui.p(model_result['message'], 
                     style="color: #dc3545; font-style: italic;")
            )
        
        life_exp_results = model_result['life_expectancy']
        happiness_results = model_result['happiness_score']
        
        return ui.div(
            ui.h5("🎯 Analysis Summary"),
            
            # Life Expectancy Results
            ui.div(
                ui.h6("💗 Life Expectancy Model", style="color: #dc3545; margin-bottom: 10px;"),
                ui.p(f"🏆 Best Model: {life_exp_results['best_model_name']}", style="margin-bottom: 5px;"),
                ui.p(f"R² Score: {life_exp_results['r2']:.3f}", style="margin-bottom: 5px;"),
                ui.p(f"RMSE: {life_exp_results['rmse']:.2f} years", style="margin-bottom: 10px;"),
                style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #dc3545;"
            ),
            
            # Happiness Score Results
            ui.div(
                ui.h6("😊 Happiness Score Model", style="color: #28a745; margin-bottom: 10px;"),
                ui.p(f"🏆 Best Model: {happiness_results['best_model_name']}", style="margin-bottom: 5px;"),
                ui.p(f"R² Score: {happiness_results['r2']:.3f}", style="margin-bottom: 5px;"),
                ui.p(f"RMSE: {happiness_results['rmse']:.2f} points", style="margin-bottom: 10px;"),
                style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #28a745;"
            ),
            
            ui.p(f"📊 Data Points: {len(model_result['data']):,}", style="text-align: center; font-weight: bold;"),
            
            style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;"
        )

    @output
    @render.ui
    def life_expectancy_plot():
        model_result = regression_model()
        
        if model_result is None:
            fig = px.scatter(title="Life Expectancy Prediction")
            fig.add_annotation(text="Click 'Run Regression Analysis' to generate plot", 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if 'error' in model_result:
            fig = px.scatter(title="Life Expectancy Prediction")
            fig.add_annotation(text=model_result['message'], 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        life_exp_results = model_result['life_expectancy']
        reg_data = life_exp_results['data']
        
        y = life_exp_results['y']
        y_pred = life_exp_results['y_pred']
        
        # Create predicted vs actual plot
        fig = px.scatter(x=y, y=y_pred, 
                        title="Predicted vs Actual Life Expectancy",
                        labels={'x': 'Actual Life Expectancy', 'y': 'Predicted Life Expectancy'},
                        hover_data={'Country': reg_data['Country'], 'Year': reg_data['Year']})
        
        # Add perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                       mode='lines', name='Perfect Prediction', 
                       line=dict(color='red', dash='dash'))
        
        fig.update_layout(
            width=800,
            height=400
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.ui
    def happiness_prediction_plot():
        model_result = regression_model()
        
        if model_result is None:
            fig = px.scatter(title="Happiness Score Prediction")
            fig.add_annotation(text="Click 'Run Regression Analysis' to generate plot", 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if 'error' in model_result:
            fig = px.scatter(title="Happiness Score Prediction")
            fig.add_annotation(text=model_result['message'], 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        happiness_results = model_result['happiness_score']
        reg_data = happiness_results['data']
        
        y = happiness_results['y']
        y_pred = happiness_results['y_pred']
        
        # Create predicted vs actual plot
        fig = px.scatter(x=y, y=y_pred, 
                        title="Predicted vs Actual Happiness Score",
                        labels={'x': 'Actual Happiness Score', 'y': 'Predicted Happiness Score'},
                        hover_data={'Country': reg_data['Country'], 'Year': reg_data['Year']})
        
        # Add perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        fig.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                       mode='lines', name='Perfect Prediction', 
                       line=dict(color='red', dash='dash'))
        
        fig.update_layout(
            width=800,
            height=400
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    @output
    @render.ui
    def feature_importance():
        model_result = regression_model()
        
        if model_result is None:
            fig = px.bar(title="Feature Importance")
            fig.add_annotation(text="Click 'Run Regression Analysis' to see feature importance", 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        if 'error' in model_result:
            fig = px.bar(title="Feature Importance")
            fig.add_annotation(text=model_result['message'], 
                              showarrow=False, font=dict(size=16))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get importance data for both targets
        life_exp_importance = model_result['life_expectancy']['importance_df'].copy()
        happiness_importance = model_result['happiness_score']['importance_df'].copy()
        
        # Create a combined dataframe for grouped bar chart
        factors = life_exp_importance['Factor'].tolist()
        
        # Prepare data for grouped bar chart
        combined_data = []
        for factor in factors:
            # Life expectancy importance
            life_imp = life_exp_importance[life_exp_importance['Factor'] == factor]['Importance'].iloc[0]
            life_coef = life_exp_importance[life_exp_importance['Factor'] == factor]['Coefficient'].iloc[0]
            
            # Happiness importance
            happiness_imp = happiness_importance[happiness_importance['Factor'] == factor]['Importance'].iloc[0]
            happiness_coef = happiness_importance[happiness_importance['Factor'] == factor]['Coefficient'].iloc[0]
            
            combined_data.append({
                'Factor': factor,
                'Life_Expectancy_Importance': life_imp,
                'Happiness_Score_Importance': happiness_imp,
                'Life_Expectancy_Impact': 'Positive' if life_coef > 0 else 'Negative',
                'Happiness_Score_Impact': 'Positive' if happiness_coef > 0 else 'Negative'
            })
        
        combined_df = pd.DataFrame(combined_data)
        
        # Create grouped bar chart using plotly
        fig = px.bar()
        
        # Add Life Expectancy bars
        fig.add_bar(
            x=combined_df['Factor'],
            y=combined_df['Life_Expectancy_Importance'],
            name='Life Expectancy',
            marker_color='#e74c3c',  # Red color for life expectancy
            text=[f"{val:.3f}" for val in combined_df['Life_Expectancy_Importance']],
            textposition='outside'
        )
        
        # Add Happiness Score bars
        fig.add_bar(
            x=combined_df['Factor'],
            y=combined_df['Happiness_Score_Importance'],
            name='Happiness Score',
            marker_color='#2ecc71',  # Green color for happiness
            text=[f"{val:.3f}" for val in combined_df['Happiness_Score_Importance']],
            textposition='outside'
        )
        
        # Update layout for better appearance
        fig.update_layout(
            title="Feature Importance Comparison: Life Expectancy vs Happiness Score",
            xaxis_title="Factors",
            yaxis_title="Importance Score",
            barmode='group',  # This creates grouped bars
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            width=1000,
            height=600,
            font=dict(size=12)
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

    # Correlation Analysis - Updated for dual targets
    @reactive.Calc
    def correlation_analysis():
        selected_factors = input.regression_factors()
        
        if not selected_factors:
            return None
        
        try:
            from scipy.stats import pearsonr, spearmanr
            import pandas as pd
            
            # Analyze correlations for both targets
            target_variables = ['Life Expectancy', 'Happiness Score']
            required_columns = target_variables + list(selected_factors)
            corr_data = data.dropna(subset=required_columns)
            
            # Remove 0 values
            for col in required_columns:
                if col in corr_data.columns:
                    corr_data = corr_data[corr_data[col] > 0]
            
            if len(corr_data) < 10:
                return None
            
            all_correlations = {}
            
            for target in target_variables:
                correlations = []
                for factor in selected_factors:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(corr_data[target], corr_data[factor])
                    # Spearman correlation (for non-linear relationships)
                    spearman_r, spearman_p = spearmanr(corr_data[target], corr_data[factor])
                    
                    correlations.append({
                        'Factor': factor,
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'Spearman_r': spearman_r,
                        'Spearman_p': spearman_p,
                        'Significance': 'Significant' if pearson_p < 0.05 else 'Not Significant'
                    })
                
                all_correlations[target] = pd.DataFrame(correlations)
            
            return {
                'life_expectancy': all_correlations['Life Expectancy'],
                'happiness_score': all_correlations['Happiness Score']
            }
            
        except ImportError:
            return {'error': 'scipy not available'}
        except Exception as e:
            return {'error': str(e)}

    @output
    @render.ui
    def correlation_results():
        corr_result = correlation_analysis()
        
        if corr_result is None:
            return ui.div(
                ui.p("Select factors to see correlation analysis", 
                     style="color: #666; font-style: italic;")
            )
        
        if 'error' in corr_result:
            return ui.div(
                ui.p(str(corr_result['error']), 
                     style="color: #dc3545; font-style: italic;")
            )
        
        life_exp_corr = corr_result['life_expectancy']
        happiness_corr = corr_result['happiness_score']
        
        # Display correlation results for both targets
        def create_correlation_section(corr_df, target_name, color):
            rows = []
            for _, row in corr_df.iterrows():
                significance_icon = "✅" if row['Significance'] == "Significant" else "❌"
                rows.append(
                    ui.div(
                        ui.p(f"{significance_icon} {row['Factor']}: ρ = {row['Pearson_r']:.3f} (p = {row['Pearson_p']:.3f})", 
                             style="margin-bottom: 4px;"),
                        style="font-size: 0.9em; color: #333;"
                    )
                )
            return ui.div(
                ui.h6(f"📊 {target_name}"),
                *rows,
                style=f"background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid {color}; margin-bottom: 15px;"
            )
        
        return ui.div(
            ui.h5("🔗 Correlation Analysis Results"),
            create_correlation_section(life_exp_corr, "Life Expectancy Correlations", "#dc3545"),
            create_correlation_section(happiness_corr, "Happiness Score Correlations", "#28a745"),
            
            ui.div(
                ui.h6("📋 Interpretation Guide:"),
                ui.div(
                    ui.p("• ✅ Significant correlations (p < 0.05) indicate reliable relationships", style="margin-bottom: 5px;"),
                    ui.p("• ❌ Non-significant correlations may be due to chance", style="margin-bottom: 5px;"),
                    ui.p("• Strong correlations: |ρ| > 0.7", style="margin-bottom: 5px;"),
                    ui.p("• Moderate correlations: 0.3 < |ρ| < 0.7", style="margin-bottom: 5px;"),
                    ui.p("• Weak correlations: |ρ| < 0.3", style="margin-bottom: 0px;")
                ),
                style="background-color: #e9ecef; padding: 15px; border-radius: 8px; margin-top: 15px;"
            )
        )

app = App(app_ui, server)