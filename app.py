from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os

# Load the dataset
data = pd.read_csv('data/all_with_happiness.csv')

# Add missing columns that might be needed for the analysis but don't exist
missing_columns = {
    'Region': 'Unknown',
    'Income Group': 'Unknown', 
    'Social Support': np.nan,
    'Freedom': np.nan,
    'Generosity': np.nan
}

for col_name, default_value in missing_columns.items():
    if col_name not in data.columns:
        data[col_name] = default_value

# Generate factor choices dynamically from numeric columns
# Exclude identifier and categorical columns
exclude_columns = ['Country', 'Year', 'Region', 'Income Group', 'Code', 'Year_Code', 'Life Expectancy', 'Life Expectancy Male', 'Life Expectancy Female']
factor_choices = [col for col in data.columns.tolist() if col not in exclude_columns]

print(f"Factor choices: {factor_choices}")

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
                                  value=[min(year_choices), max(year_choices)]),
                    ui.input_checkbox("show_average", "Show Average Over Range", value=True),
                    ui.input_select("color_scale", "Color Scale", 
                                  choices=["Viridis", "Plasma", "Blues", "Reds", "Greens"],
                                  selected="Viridis"),
                    ui.hr(),
                    ui.h4("Country Details"),
                    ui.input_select("country1", "Select a Country", choices=country_choices),
                    ui.output_ui("country_details")
                ),
                ui.column(9,
                    ui.h3("Life Expectancy World Map"),
                    ui.output_ui("world_map_plot"),
                    ui.p("The map shows life expectancy, happiness score, and GDP (Million $) across different countries. Hover over a country to see details.")
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
                    # Show both inputs but only use the relevant one in the server function
                    ui.input_select("country2", "Select a Country", choices=country_choices),
                    ui.input_select("continent2", "Select a Continent", choices=continent_choices),
                    ui.input_slider("years_range2", "Year Range", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=[min(year_choices), max(year_choices)]),
                    ui.hr(),
                    ui.input_radio_buttons("plot_type2", "Plot Type", 
                                         ["Line Chart", "Bar Chart", "Scatter Plot"])
                ),
                ui.column(9,
                    ui.h3("Happiness, GDP, and Life Expectancy Trends"),
                    ui.output_ui("happiness_plot")
                )
            )
        ),

        # Tab 3 - Factor Indicators
        ui.nav_panel("Factor Indicators", 
            ui.row(
                ui.column(3,
                    ui.h4("Controls"),
                    ui.input_select("view_type3", "View Type", 
                                  choices=["Country View", "Continent View"],
                                  selected="Country View"),
                    ui.input_select("country3", "Select a Country", choices=country_choices),
                    ui.input_select("continent3", "Select a Continent", choices=continent_choices),
                    ui.input_slider("years_range3", "Year Range", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=[min(year_choices), max(year_choices)]),
                    ui.hr(),
                    ui.input_selectize("factors3", "Select Factors to Visualize", 
                                     choices=factor_choices,
                                     multiple=True,
                                     selected=None),
                    ui.hr(),
                ),
                ui.column(9,
                    ui.h3("Time Series Analysis: Factors Affecting Life Expectancy"),
                    ui.output_ui("economic_plot"),
                    ui.hr(),
                    ui.h4("Spider Charts: Factor Impact on Life Expectancy"),
                    ui.row(
                        ui.column(12, ui.output_ui("spider_total")),
                    ),
                    ui.h4(""),
                    ui.row(
                        ui.column(6, ui.output_ui("spider_male")),
                        ui.column(6, ui.output_ui("spider_female"))
                    )
                )
            )
        ),
        
        # Tab 4 - Comparative Analysis
        ui.nav_panel("Life Expectancy Affected by Multiple Factors", 
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
                                   value=[min(year_choices), max(year_choices)]),
                    ui.hr(),
                    ui.input_selectize("factors4", "Select Factors to Visualize", 
                                     choices=factor_choices,
                                     multiple=True,
                                     selected=None),
                    ui.hr(),
                    ui.output_ui("selected_countries_list")
                ),
                ui.column(9,
                    ui.h3("Time Series Analysis: Top/Bottom Countries by Life Expectancy"),
                    ui.output_ui("comparison_time_series"),
                    ui.hr(),
                    ui.h4("Spider Charts: Factor Impact Comparison"),
                    ui.row(
                        ui.column(12, ui.output_ui("comparison_spider_total")),
                    ),
                    ui.h4(""),
                    ui.row(
                        ui.column(6, ui.output_ui("comparison_spider_male")),
                        ui.column(6, ui.output_ui("comparison_spider_female"))
                    )
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
        show_average = input.show_average()
        
        # Filter data for the selected year range
        year_data = data[(data['Year'] >= year_min) & (data['Year'] <= year_max)]
        
        # Check if we have any data to display
        if year_data.empty:
            print("Year data is empty")
            # Create an empty map with a message if no data available
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
                categorical_cols = ['Region', 'Income Group']
                # print(year_data.columns)
                # First handle numeric columns with mean
                avg_numeric = year_data.groupby('Country')[numeric_cols].mean()
                
                # Then handle categorical columns by taking first non-null value
                # Use a lambda function to prevent string concatenation issues
                avg_categorical = year_data.groupby('Country').agg({
                    col: lambda x: x.iloc[0] if len(x) > 0 else None 
                    for col in categorical_cols
                })
                
                # Combine the results
                avg_data = pd.concat([avg_numeric, avg_categorical], axis=1).reset_index()
                
                # Create title with range information
                title = f"Average Life Expectancy by Country ({int(year_min)}-{int(year_max)})"
                plot_data = avg_data
            except Exception as e:
                # Fallback to using the most recent year if aggregation fails
                print(f"Error in aggregation: {e}")  # For debugging
                title = f"Life Expectancy by Country ({int(year_max)}) - Showing latest year"
                plot_data = year_data[year_data['Year'] == year_max]
        else:
            # Use the most recent year in the range
            title = f"Life Expectancy by Country ({int(year_max)})"
            plot_data = year_data[year_data['Year'] == year_max]
            
            # If no data for the most recent year, use the last available year
            if plot_data.empty and not year_data.empty:
                last_available_year = year_data['Year'].max()
                plot_data = year_data[year_data['Year'] == last_available_year]
                title = f"Life Expectancy by Country ({int(last_available_year)}) - Most recent available data"

        # Create a choropleth map
        fig = px.choropleth(
            plot_data,
            locations="Country",  # country names in dataset
            locationmode="country names",  # set of locations match entries in `locations`
            color="Life Expectancy",  # value in column 'Life Expectancy' determines color
            hover_name="Country",  # column to add to hover information
            color_continuous_scale=input.color_scale().lower(),
            title=title,
            hover_data={
                "Life Expectancy": ":.1f",
                "Happiness Score": ":.2f", 
                "GDP per Capita": ":,.0f"
            } if all(col in plot_data.columns for col in ["Life Expectancy", "Happiness Score", "GDP per Capita"]) else None,
            projection="natural earth"  # map projection type
        )
        
        # Improve layout
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            margin={"r":0,"t":50,"l":0,"b":0},
            coloraxis_colorbar={
                'title': 'Life Expectancy (years)'
            },
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
                numeric_data = country_data[columns_to_average]
                
                # Calculate averages
                avg_data = numeric_data.mean(skipna=True)

                life_exp = avg_data['Life Expectancy']
                happiness = avg_data['Happiness Score']
                gdp = avg_data['GDP per Capita']
                
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
        
    # Tab 2 - Happiness Plot  
    @output
    @render.ui
    def happiness_plot():
        view_type = input.view_type()
        year_min, year_max = input.years_range2()
        plot_type = input.plot_type2()
        
        if view_type == "Country View":
            # Original country view functionality
            selected_country = input.country2()
            
            # Filter data for the selected country and years
            filtered_data = data[(data['Country'] == selected_country) & 
                               (data['Year'] >= year_min) & 
                               (data['Year'] <= year_max)]
            
            if filtered_data.empty:
                # Return an empty plot with a message
                fig = px.scatter(title=f"No data available for {selected_country}")
                fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
            
            # Normalize the data for better visualization (scale to 0-100)
            normalized_data = filtered_data.copy()
            normalized_data['Happiness Score Normalized'] = (filtered_data['Happiness Score'] / 10) * 100
            normalized_data['GDP per Capita Normalized'] = (filtered_data['GDP per Capita'] / filtered_data['GDP per Capita'].max()) * 100
            normalized_data['Life Expectancy Normalized'] = (filtered_data['Life Expectancy'] / filtered_data['Life Expectancy'].max()) * 100
            
            if plot_type == "Line Chart":
                # Create multi-line chart
                fig = px.line(title=f"Multi-Factor Analysis for {selected_country} ({year_min}-{year_max})")
                
                # Add three lines
                fig.add_scatter(x=normalized_data['Year'], y=normalized_data['Happiness Score Normalized'], 
                              mode='lines+markers', name='Happiness Score', line=dict(color='blue'))
                fig.add_scatter(x=normalized_data['Year'], y=normalized_data['GDP per Capita Normalized'], 
                              mode='lines+markers', name='GDP per Capita', line=dict(color='green'))
                fig.add_scatter(x=normalized_data['Year'], y=normalized_data['Life Expectancy Normalized'], 
                              mode='lines+markers', name='Life Expectancy', line=dict(color='red'))
                
                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Normalized Scale (0-100)",
                    yaxis=dict(range=[0, 100]),
                    hovermode="x unified"
                )
                
            elif plot_type == "Bar Chart":
                # Reshape data for grouped bar chart
                import pandas as pd
                melted_data = pd.melt(normalized_data, 
                                    id_vars=['Year'], 
                                    value_vars=['Happiness Score Normalized', 'GDP per Capita Normalized', 'Life Expectancy Normalized'],
                                    var_name='Metric', value_name='Normalized Value')
                
                fig = px.bar(melted_data, x="Year", y="Normalized Value", color="Metric",
                           title=f"Multi-Factor Analysis for {selected_country} ({year_min}-{year_max})",
                           barmode='group')
                
                fig.update_layout(yaxis=dict(range=[0, 100]))
                
            else:  # Scatter Plot
                # Create scatter plot matrix with cleaner names
                scatter_data = normalized_data.copy()
                scatter_data['Happiness Score'] = scatter_data['Happiness Score Normalized']
                scatter_data['GDP per Capita'] = scatter_data['GDP per Capita Normalized']
                scatter_data['Life Expectancy'] = scatter_data['Life Expectancy Normalized']
                
                fig = px.scatter_matrix(scatter_data, 
                                      dimensions=["Happiness Score", "GDP per Capita", "Life Expectancy"],
                                      title=f"Correlation Matrix for {selected_country} ({year_min}-{year_max})")
                
                fig.update_layout(
                    xaxis=dict(range=[0, 100]),
                    yaxis=dict(range=[0, 100]),
                    xaxis2=dict(range=[0, 100]),
                    yaxis2=dict(range=[0, 100]),
                    xaxis3=dict(range=[0, 100]),
                    yaxis3=dict(range=[0, 100])
                )

        else:
            # Continent view - calculate average for countries in the continent
            selected_continent = input.continent2()
            
            # Get countries in the selected continent
            if selected_continent in continents_data:
                continent_countries = continents_data[selected_continent]
                
                # Filter for countries in the continent and the selected years
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                
                if continent_data.empty:
                    # Return an empty plot with a message
                    fig = px.scatter(title=f"No data available for countries in {selected_continent}")
                    fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                # Calculate average by year for the continent - use all numeric columns
                agg_dict = {col: 'mean' for col in factor_choices if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                
                # Normalize the data
                plot_data['Happiness Score Normalized'] = (plot_data['Happiness Score'] / 10) * 100
                plot_data['GDP per Capita Normalized'] = (plot_data['GDP per Capita'] / plot_data['GDP per Capita'].max()) * 100
                plot_data['Life Expectancy Normalized'] = (plot_data['Life Expectancy'] / plot_data['Life Expectancy'].max()) * 100
                
                if plot_type == "Line Chart":
                    # Create multi-line chart
                    fig = px.line(title=f"Multi-Factor Analysis for {selected_continent} ({year_min}-{year_max})")
                    
                    # Add three lines
                    fig.add_scatter(x=plot_data['Year'], y=plot_data['Happiness Score Normalized'], 
                                  mode='lines+markers', name='Happiness Score', line=dict(color='blue'))
                    fig.add_scatter(x=plot_data['Year'], y=plot_data['GDP per Capita Normalized'], 
                                  mode='lines+markers', name='GDP per Capita', line=dict(color='green'))
                    fig.add_scatter(x=plot_data['Year'], y=plot_data['Life Expectancy Normalized'], 
                                  mode='lines+markers', name='Life Expectancy', line=dict(color='red'))
                    
                    fig.update_layout(
                        xaxis_title="Year",
                        yaxis_title="Normalized Scale (0-100)",
                        yaxis=dict(range=[0, 100]),
                        hovermode="x unified"
                    )
                    
                elif plot_type == "Bar Chart":
                    # Reshape data for grouped bar chart
                    import pandas as pd
                    melted_data = pd.melt(plot_data, 
                                        id_vars=['Year'], 
                                        value_vars=['Happiness Score Normalized', 'GDP per Capita Normalized', 'Life Expectancy Normalized'],
                                        var_name='Metric', value_name='Normalized Value')
                    
                    fig = px.bar(melted_data, x="Year", y="Normalized Value", color="Metric",
                               title=f"Multi-Factor Analysis for {selected_continent} ({year_min}-{year_max})",
                               barmode='group')
                    
                    fig.update_layout(yaxis=dict(range=[0, 100]))
                
                else:  # Scatter Plot
                    # Create scatter plot matrix with cleaner names
                    scatter_data = plot_data.copy()
                    scatter_data['Happiness Score'] = scatter_data['Happiness Score Normalized']
                    scatter_data['GDP per Capita'] = scatter_data['GDP per Capita Normalized']
                    scatter_data['Life Expectancy'] = scatter_data['Life Expectancy Normalized']
                    
                    fig = px.scatter_matrix(scatter_data, 
                                          dimensions=["Happiness Score", "GDP per Capita", "Life Expectancy"],
                                          title=f"Correlation Matrix for {selected_continent} ({year_min}-{year_max})")
            
                    # Set axis ranges
                    fig.update_layout(
                        xaxis=dict(range=[0, 100]),
                        yaxis=dict(range=[0, 100]),
                        xaxis2=dict(range=[0, 100]),
                        yaxis2=dict(range=[0, 100]),
                        xaxis3=dict(range=[0, 100]),
                        yaxis3=dict(range=[0, 100])
                    )
            
            else:
                # Return an empty plot with a message for invalid continent
                fig = px.scatter(title=f"No data available for {selected_continent}")
                fig.add_annotation(text="Invalid continent selection", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
        # Add better labels and formatting
        if plot_type != "Scatter Plot":
            fig.update_layout(
                hovermode="closest",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        else:
            # For scatter plots, add height and width separately
            fig.update_layout(
                height=700,  # Scatter matrix needs more height
                width=900
            )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
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
        
        if view_type == "Country View":
            selected_country = input.country3()
            
            # Filter data for the selected country and years
            filtered_data = data[(data['Country'] == selected_country) & 
                               (data['Year'] >= year_min) & 
                               (data['Year'] <= year_max)]
            
            if filtered_data.empty:
                fig = px.scatter(title=f"No data available for {selected_country}")
                fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
            
            title_location = selected_country
            plot_data = filtered_data
            
        else:  # Continent View
            selected_continent = input.continent3()
            
            if selected_continent in continents_data:
                continent_countries = continents_data[selected_continent]
                
                # Filter for countries in the continent and the selected years
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                
                if continent_data.empty:
                    fig = px.scatter(title=f"No data available for countries in {selected_continent}")
                    fig.add_annotation(text="No data available", showarrow=False, font=dict(size=20))
                    return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
                # Calculate average by year for the continent
                agg_dict = {col: 'mean' for col in factor_choices if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                
                title_location = selected_continent
            else:
                fig = px.scatter(title=f"No data available for {selected_continent}")
                fig.add_annotation(text="Invalid continent selection", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Create time series line plot
        fig = px.line(title=f"Time Series Analysis for {title_location} ({year_min}-{year_max})")
        
        # Add line for Life Expectancy (always shown as reference)
        fig.add_scatter(x=plot_data['Year'], y=plot_data['Life Expectancy'], 
                      mode='lines+markers', name='Life Expectancy (Reference)', 
                      line=dict(color='red', width=3))
        
        # Color palette for factors
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Add lines for selected factors
        for i, factor in enumerate(selected_factors):
            if factor in plot_data.columns:
                # Normalize factor to same scale as life expectancy for better comparison
                if factor == 'Happiness Score':
                    # Scale happiness score (0-10) to life expectancy range
                    normalized_values = plot_data[factor] * (plot_data['Life Expectancy'].max() / 10)
                elif factor == 'GDP per Capita':
                    # Scale GDP to life expectancy range
                    normalized_values = (plot_data[factor] / plot_data[factor].max()) * plot_data['Life Expectancy'].max()
                else:
                    # For other factors, scale to life expectancy range
                    normalized_values = (plot_data[factor] / plot_data[factor].max()) * plot_data['Life Expectancy'].max()
                
                line_style = dict(color=colors[i % len(colors)], width=2)
                
                # Remove special highlighting for affective factor
                fig.add_scatter(x=plot_data['Year'], y=normalized_values, 
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
        
        # Remove the annotation about most affective factor
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
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
    
    # Tab 4 - Time Series Comparison
    @output
    @render.ui
    def comparison_time_series():
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
        
        # Calculate average across all selected countries by year
        agg_dict = {col: 'mean' for col in factor_choices if col in filtered_data.columns}
        plot_data = filtered_data.groupby('Year').agg(agg_dict).reset_index()
        
        title_text = f"Average for {rank_type.replace('Life Expectancy', '').strip()} {num_countries} Countries"
        
        # Create time series line plot
        fig = px.line(title=f"Time Series Analysis: {title_text} ({year_min}-{year_max})")
        
        # Add line for Life Expectancy (always shown as reference)
        fig.add_scatter(x=plot_data['Year'], y=plot_data['Life Expectancy'], 
                      mode='lines+markers', name='Life Expectancy (Reference)', 
                      line=dict(color='red', width=3))
        
        # Color palette for factors
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Add lines for selected factors
        for i, factor in enumerate(selected_factors):
            if factor in plot_data.columns:
                # Normalize factor to same scale as life expectancy for better comparison
                if factor == 'Happiness Score':
                    # Scale happiness score (0-10) to life expectancy range
                    normalized_values = plot_data[factor] * (plot_data['Life Expectancy'].max() / 10)
                elif factor == 'GDP per Capita':
                    # Scale GDP to life expectancy range
                    normalized_values = (plot_data[factor] / plot_data[factor].max()) * plot_data['Life Expectancy'].max()
                else:
                    # For other factors, scale to life expectancy range
                    normalized_values = (plot_data[factor] / plot_data[factor].max()) * plot_data['Life Expectancy'].max()
                
                line_style = dict(color=colors[i % len(colors)], width=2)
                
                fig.add_scatter(x=plot_data['Year'], y=normalized_values, 
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
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    # Tab 4 - Spider Charts
    @output
    @render.ui
    def comparison_spider_total():
        countries = get_ranked_countries()
        selected_factors = input.factors4()
        year_min, year_max = input.years_range4()
        rank_type = input.rank_type()
        num_countries = input.num_countries()
        
        if not countries or not selected_factors:
            fig = px.scatter(title="Total Life Expectancy Comparison")
            fig.add_annotation(text="Select countries and factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data
        filtered_data = data[(data['Country'].isin(countries)) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
        
        if filtered_data.empty:
            fig = px.scatter(title="Total Life Expectancy Comparison")
            fig.add_annotation(text="No data available", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Calculate average across countries and years
        agg_dict = {col: 'mean' for col in factor_choices + ['Life Expectancy'] if col in filtered_data.columns}
        plot_data = filtered_data.agg(agg_dict).to_frame().T
        plot_data['Year'] = year_max  # Add year for compatibility with helper function
        
        title_text = f"{rank_type.replace('Life Expectancy', '').strip()} {num_countries} Countries"
        fig = create_spider_chart('Life Expectancy', f"Both male and female - {title_text}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def comparison_spider_male():
        countries = get_ranked_countries()
        selected_factors = input.factors4()
        year_min, year_max = input.years_range4()
        rank_type = input.rank_type()
        num_countries = input.num_countries()
        
        if not countries or not selected_factors:
            fig = px.scatter(title="Male Life Expectancy Comparison")
            fig.add_annotation(text="Select countries and factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data
        filtered_data = data[(data['Country'].isin(countries)) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
        
        if filtered_data.empty:
            fig = px.scatter(title="Male Life Expectancy Comparison")
            fig.add_annotation(text="No data available", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Calculate average across countries and years
        male_col = 'Life Expectancy Male' if 'Life Expectancy Male' in filtered_data.columns else None
        agg_dict = {col: 'mean' for col in factor_choices + ([male_col] if male_col else []) if col in filtered_data.columns}
        plot_data = filtered_data.agg(agg_dict).to_frame().T
        plot_data['Year'] = year_max  # Add year for compatibility with helper function
        
        title_text = f"{rank_type.replace('Life Expectancy', '').strip()} {num_countries} Countries"
        fig = create_spider_chart('Life Expectancy Male', f"Male - {title_text}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def comparison_spider_female():
        countries = get_ranked_countries()
        selected_factors = input.factors4()
        year_min, year_max = input.years_range4()
        rank_type = input.rank_type()
        num_countries = input.num_countries()
        
        if not countries or not selected_factors:
            fig = px.scatter(title="Female Life Expectancy Comparison")
            fig.add_annotation(text="Select countries and factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data
        filtered_data = data[(data['Country'].isin(countries)) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
        
        if filtered_data.empty:
            fig = px.scatter(title="Female Life Expectancy Comparison")
            fig.add_annotation(text="No data available", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Calculate average across countries and years
        female_col = 'Life Expectancy Female' if 'Life Expectancy Female' in filtered_data.columns else None
        agg_dict = {col: 'mean' for col in factor_choices + ([female_col] if female_col else []) if col in filtered_data.columns}
        plot_data = filtered_data.agg(agg_dict).to_frame().T
        plot_data['Year'] = year_max  # Add year for compatibility with helper function
        
        title_text = f"{rank_type.replace('Life Expectancy', '').strip()} {num_countries} Countries"
        fig = create_spider_chart('Life Expectancy Female', f"Female - {title_text}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    # Spider Charts for Life Expectancy Types
    def create_spider_chart(life_exp_col, chart_title, selected_factors, plot_data):
        """Helper function to create spider charts"""
        if life_exp_col not in plot_data.columns or plot_data[life_exp_col].isna().all():
            fig = px.scatter(title=f"{chart_title} - No Data Available")
            fig.add_annotation(text="No data", showarrow=False, font=dict(size=14))
            return fig
        
        # Get the latest year data for spider chart
        latest_year = plot_data['Year'].max()
        latest_data = plot_data[plot_data['Year'] == latest_year]
        
        if latest_data.empty:
            fig = px.scatter(title=f"{chart_title} - No Data Available")
            fig.add_annotation(text="No data", showarrow=False, font=dict(size=14))
            return fig
        
        # Prepare data for spider chart
        spider_data = []
        
        # Add life expectancy
        life_exp_value = latest_data[life_exp_col].iloc[0]
        
        # Add selected factors (normalized to life expectancy scale)
        for factor in selected_factors:
            if factor in latest_data.columns:
                factor_value = latest_data[factor].iloc[0]
                
                # Replace NaN with mean from the entire plot_data
                if pd.isna(factor_value):
                    factor_value = plot_data[factor].mean()
                    # If mean is still NaN (all values are NaN), skip this factor
                    if pd.isna(factor_value):
                        continue
                
                # Normalize to life expectancy scale
                if factor == 'Happiness Score':
                    normalized_value = (factor_value / 10) * life_exp_value
                elif factor == 'GDP per Capita':
                    max_gdp = plot_data[factor].max()
                    normalized_value = (factor_value / max_gdp) * life_exp_value if max_gdp > 0 else 0
                else:
                    max_factor = plot_data[factor].max()
                    normalized_value = (factor_value / max_factor) * life_exp_value if max_factor > 0 else 0
                
                spider_data.append({
                    'factor': factor,
                    'value': factor_value,
                    'normalized': normalized_value
                })

        if len(spider_data) < 2:
            fig = px.scatter(title=f"{chart_title} - Insufficient Data")
            fig.add_annotation(text="Need more factors", showarrow=False, font=dict(size=14))
            return fig
        
        # Create spider chart
        import plotly.graph_objects as go
        
        categories = [item['factor'] for item in spider_data]
        values = [item['normalized'] for item in spider_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=chart_title.split(' - ')[0],
            line_color='rgba(255, 0, 0, 0.8)',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 100]
                )),
            showlegend=False,
            title=dict(text=chart_title, x=0.5, font=dict(size=14)),
            # height=350,
            # width=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @output
    @render.ui
    def spider_total():
        view_type = input.view_type3()
        year_min, year_max = input.years_range3()
        selected_factors = input.factors3()
        
        if not selected_factors:
            fig = px.scatter(title="Total Life Expectancy")
            fig.add_annotation(text="Select factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data (same logic as economic_plot)
        if view_type == "Country View":
            selected_country = input.country3()
            plot_data = data[(data['Country'] == selected_country) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
            title_location = selected_country
        else:
            selected_continent = input.continent3()
            if selected_continent in continents_data:
                continent_countries = continents_data[selected_continent]
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                agg_dict = {col: 'mean' for col in factor_choices + ['Life Expectancy'] if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                title_location = selected_continent
            else:
                plot_data = pd.DataFrame()

        fig = create_spider_chart('Life Expectancy', f"Both male and female - {title_location}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def spider_male():
        view_type = input.view_type3()
        year_min, year_max = input.years_range3()
        selected_factors = input.factors3()
        
        if not selected_factors:
            fig = px.scatter(title="Male Life Expectancy")
            fig.add_annotation(text="Select factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data (same logic as economic_plot)
        if view_type == "Country View":
            selected_country = input.country3()
            plot_data = data[(data['Country'] == selected_country) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
            title_location = selected_country
        else:
            selected_continent = input.continent3()
            if selected_continent in continents_data:
                continent_countries = continents_data[selected_continent]
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                # Include male life expectancy in aggregation if it exists
                male_col = 'Life Expectancy Male' if 'Life Expectancy Male' in continent_data.columns else None
                agg_dict = {col: 'mean' for col in factor_choices + ([male_col] if male_col else []) if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                title_location = selected_continent
            else:
                plot_data = pd.DataFrame()
        
        fig = create_spider_chart('Life Expectancy Male', f"Male - {title_location}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
    @output
    @render.ui
    def spider_female():
        view_type = input.view_type3()
        year_min, year_max = input.years_range3()
        selected_factors = input.factors3()
        
        if not selected_factors:
            fig = px.scatter(title="Female Life Expectancy")
            fig.add_annotation(text="Select factors first", showarrow=False, font=dict(size=12))
            return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
        # Get plot data (same logic as economic_plot)
        if view_type == "Country View":
            selected_country = input.country3()
            plot_data = data[(data['Country'] == selected_country) & 
                           (data['Year'] >= year_min) & 
                           (data['Year'] <= year_max)]
            title_location = selected_country
        else:
            selected_continent = input.continent3()
            if selected_continent in continents_data:
                continent_countries = continents_data[selected_continent]
                continent_data = data[
                    (data['Country'].isin(continent_countries)) & 
                    (data['Year'] >= year_min) & 
                    (data['Year'] <= year_max)
                ]
                # Include female life expectancy in aggregation if it exists
                female_col = 'Life Expectancy Female' if 'Life Expectancy Female' in continent_data.columns else None
                agg_dict = {col: 'mean' for col in factor_choices + ([female_col] if female_col else []) if col in continent_data.columns}
                plot_data = continent_data.groupby('Year').agg(agg_dict).reset_index()
                title_location = selected_continent
            else:
                plot_data = pd.DataFrame()
        
        fig = create_spider_chart('Life Expectancy Female', f"Female - {title_location}", selected_factors, plot_data)
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
    
app = App(app_ui, server)