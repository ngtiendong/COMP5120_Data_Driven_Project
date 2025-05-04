from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os

# Load the life expectancy data
try:
    data = pd.read_csv('data/life_expectancy.csv')
except FileNotFoundError:
    # If the file doesn't exist yet, create a placeholder dataframe
    data = pd.DataFrame({
        'Country': ['Sample Country'],
        'Life Expectancy': [75.0],
        'Happiness Score': [7.0],
        'GDP per Capita': [10000]
    })

# Load continents data
try:
    with open('data/config/continents.json', 'r') as f:
        continents_data = json.load(f)
    continent_choices = list(continents_data.keys())
except FileNotFoundError:
    continents_data = {}
    continent_choices = []

# Convert country names from NumPy array to list
country_choices = list(data['Country'].unique())
year_choices = list(data['Year'].unique()) if 'Year' in data.columns else [2019]

# Define UI with tabs
app_ui = ui.page_fluid(
    ui.include_css("www/styles.css"),
    ui.panel_title("Life Expectancy, Happiness, and Economic Analysis"),
    
    # Horizontal tabs on top
    ui.navset_tab(
        # Tab 1 - Life Expectancy Overview with World Map
        ui.nav_panel("Life Expectancy Overview", 
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
                    ui.output_text("country_details")
                ),
                ui.column(9,
                    ui.h3("Life Expectancy World Map"),
                    ui.output_ui("world_map_plot"),
                    ui.p("The map shows life expectancy across different countries. Hover over a country to see details.")
                )
            )
        ),
        
        # Tab 2 - Happiness, GDP, and Life Expectancy Analysis
        ui.nav_panel("Happiness, GDP, and Life Expectancy Analysis", 
            ui.row(
                ui.column(4,
                    ui.h4("Happiness Index Controls"),
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
                ui.column(8,
                    ui.h3("Happiness Score Analysis"),
                    ui.output_ui("happiness_plot")
                )
            )
        ),
        
        # Tab 3 - Economic Indicators
        ui.nav_panel("Economic Indicators", 
            ui.row(
                ui.column(4,
                    ui.h4("Economic Data Controls"),
                    ui.input_select("region3", "Select Region", 
                                  ["All", "Europe", "Asia", "Africa", "Americas", "Oceania"]),
                    ui.input_numeric("gdp_threshold", "GDP Threshold (USD)", value=10000),
                    ui.hr(),
                    ui.input_checkbox("log_scale", "Use Log Scale", value=False)
                ),
                ui.column(8,
                    ui.h3("Economic Indicators and Life Expectancy \n (Scatter Plot Matrix showing factors mostly affecting life expectancy regarding socio-economic ones)"),
                    ui.output_ui("economic_plot")
                )
            )
        ),
        
        # Tab 4 - Comparative Analysis
        ui.nav_panel("Life Expectancy Affected by Multiple Factors", 
            ui.row(
                ui.column(4,
                    ui.h4("Comparison Controls"),
                    ui.input_selectize("countries4", "Select Countries", 
                                     choices=country_choices, multiple=True,
                                     selected=country_choices[:3] if len(country_choices) >= 3 else country_choices),
                    ui.input_slider("year4", "Select Year", 
                                   min=min(year_choices), max=max(year_choices), 
                                   value=max(year_choices)),
                    ui.hr(),
                    ui.input_select("color_var", "Color By", 
                                  ["Region", "Income Group"])
                ),
                ui.column(8,
                    ui.h3("Multi-Factor Comparison \n (Radar Chart showing factors in socio-economic & happiness dataset affecting life expectancy the most (5 factors per chart))"),
                    ui.output_ui("comparison_plot")
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
                title = f"Average Life Expectancy by Country ({year_min}-{year_max})"
                plot_data = avg_data
            except Exception as e:
                # Fallback to using the most recent year if aggregation fails
                print(f"Error in aggregation: {e}")  # For debugging
                title = f"Life Expectancy by Country ({year_max}) - Showing latest year"
                plot_data = year_data[year_data['Year'] == year_max]
        else:
            # Use the most recent year in the range
            title = f"Life Expectancy by Country ({year_max})"
            plot_data = year_data[year_data['Year'] == year_max]
            
            # If no data for the most recent year, use the last available year
            if plot_data.empty and not year_data.empty:
                last_available_year = year_data['Year'].max()
                plot_data = year_data[year_data['Year'] == last_available_year]
                title = f"Life Expectancy by Country ({last_available_year}) - Most recent available data"

        # Create a choropleth map
        fig = px.choropleth(
            plot_data,
            locations="Country",  # country names in dataset
            locationmode="country names",  # set of locations match entries in `locations`
            color="Life Expectancy",  # value in column 'Life Expectancy' determines color
            hover_name="Country",  # column to add to hover information
            color_continuous_scale=input.color_scale().lower(),
            title=title,
            hover_data=["Life Expectancy", "Happiness Score", "GDP per Capita"],
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
    @render.text
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
                return f"No data available for {selected_country} in selected years"
            
            if show_average:
                # Get only numeric columns we need for averaging
                columns_to_average = ['Life Expectancy', 'Happiness Score', 'GDP per Capita']
                numeric_data = country_data[columns_to_average]
                
                # Calculate averages
                avg_data = numeric_data.mean()
                
                life_exp = avg_data['Life Expectancy']
                happiness = avg_data['Happiness Score']
                gdp = avg_data['GDP per Capita']
                
                return (f"Country: {selected_country}\n"
                        f"Years: {year_min}-{year_max} (Average)\n"
                        f"Life Expectancy: {life_exp:.1f} years\n"
                        f"Happiness Score: {happiness:.2f}/10\n"
                        f"GDP per Capita: ${gdp:,.2f}")
            else:
                # Show most recent year
                recent_data = country_data[country_data['Year'] == year_max]
                
                if recent_data.empty:
                    return f"No data available for {selected_country} in {year_max}"
                
                life_exp = recent_data['Life Expectancy'].values[0]
                happiness = recent_data['Happiness Score'].values[0]
                gdp = recent_data['GDP per Capita'].values[0]
                
                return (f"Country: {selected_country}\n"
                        f"Year: {year_max}\n"
                        f"Life Expectancy: {life_exp:.1f} years\n"
                        f"Happiness Score: {happiness:.2f}/10\n"
                        f"GDP per Capita: ${gdp:,.2f}")
        except (IndexError, KeyError):
            return f"No data available for {selected_country}"
        
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
            
            if plot_type == "Line Chart":
                fig = px.line(filtered_data, x="Year", y="Happiness Score", 
                           title=f"Happiness Score for {selected_country} ({year_min}-{year_max})")
            elif plot_type == "Bar Chart":
                fig = px.bar(filtered_data, x="Year", y="Happiness Score", 
                          title=f"Happiness Score for {selected_country} ({year_min}-{year_max})")
            else:  # Scatter Plot
                fig = px.scatter(filtered_data, x="Year", y="Happiness Score", 
                              size="GDP per Capita", color="Life Expectancy",
                              title=f"Happiness Score for {selected_country} ({year_min}-{year_max})")
        else:
            # Continent view - calculate average happiness for countries in the continent
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
                
                # Calculate average happiness by year for the continent
                avg_by_year = continent_data.groupby('Year').agg({
                    'Happiness Score': 'mean',
                    'GDP per Capita': 'mean',
                    'Life Expectancy': 'mean'
                }).reset_index()
                
                if plot_type == "Line Chart":
                    fig = px.line(avg_by_year, x="Year", y="Happiness Score", 
                               title=f"Average Happiness Score for {selected_continent} ({year_min}-{year_max})")
                elif plot_type == "Bar Chart":
                    fig = px.bar(avg_by_year, x="Year", y="Happiness Score", 
                              title=f"Average Happiness Score for {selected_continent} ({year_min}-{year_max})")
                else:  # Scatter Plot
                    fig = px.scatter(avg_by_year, x="Year", y="Happiness Score", 
                                  size="GDP per Capita", color="Life Expectancy",
                                  title=f"Average Happiness Score for {selected_continent} ({year_min}-{year_max})")
            else:
                # Return an empty plot with a message for invalid continent
                fig = px.scatter(title=f"No data available for {selected_continent}")
                fig.add_annotation(text="Invalid continent selection", showarrow=False, font=dict(size=20))
                return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
                
        # Add better labels and formatting
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Happiness Score (0-10)",
            hovermode="closest"
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
    # Tab 3 - Economic Plot
    @output
    @render.ui
    def economic_plot():
        selected_region = input.region3()
        gdp_threshold = input.gdp_threshold()
        use_log = input.log_scale()
        
        # Filter by region and GDP threshold
        if selected_region == "All":
            filtered_data = data[data['GDP per Capita'] >= gdp_threshold]
        else:
            filtered_data = data[(data['Region'] == selected_region) & 
                               (data['GDP per Capita'] >= gdp_threshold)]
        
        # Get the most recent year for each country
        most_recent = filtered_data.sort_values('Year').groupby('Country').last().reset_index()
        
        # Create scatter plot
        fig = px.scatter(most_recent, 
                       x="GDP per Capita", 
                       y="Life Expectancy",
                       color="Region",
                       size="Happiness Score",
                       hover_name="Country",
                       log_x=use_log,
                       title=f"Relationship Between GDP and Life Expectancy ({selected_region})")
        
        # Add trendline
        fig.update_layout(
            xaxis_title="GDP per Capita (USD)",
            yaxis_title="Life Expectancy (years)",
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))
        
    # Tab 4 - Comparison Plot
    @output
    @render.ui
    def comparison_plot():
        selected_countries = input.countries4()
        selected_year = input.year4()
        color_var = input.color_var()
        
        # Filter by countries and year
        filtered_data = data[(data['Country'].isin(selected_countries)) & 
                           (data['Year'] == selected_year)]
        
        # Create parallel coordinates plot
        fig = px.parallel_coordinates(
            filtered_data,
            dimensions=["GDP per Capita", "Life Expectancy", "Happiness Score", 
                        "Social Support", "Freedom", "Generosity"],
            color=color_var,
            labels={
                "GDP per Capita": "GDP per Capita (USD)",
                "Life Expectancy": "Life Expectancy (years)",
                "Happiness Score": "Happiness Score",
                "Social Support": "Social Support",
                "Freedom": "Freedom",
                "Generosity": "Generosity"
            },
            title=f"Comparison of Countries ({selected_year})"
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn"))

app = App(app_ui, server)