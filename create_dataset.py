import pandas as pd

def merge_datasets():
    # Load the datasets
    all_df = pd.read_csv('data/all.csv')
    happiness_df = pd.read_csv('data/happiness-cantril-ladder.csv')
    
    # List of countries to ignore
    ignorance_country_list = [
        'West Bank and Gaza', 'St. Martin (French part)', 'Kosovo',
        'Africa Eastern and Southern', 'Africa Western and Central', 'American Samoa', 'Arab World',
        'Aruba', 'Bahamas, The', 'Bermuda', 'British Virgin Islands', 'Brunei Darussalam',
        'Caribbean small states', 'Cayman Islands', 'Central Europe and the Baltics', 'Channel Islands',
        'Congo, Dem. Rep.', 'Congo, Rep.', 'Cote d\'Ivoire', 'Curacao', 'Czechia', 'Early-demographic dividend',
        'East Asia & Pacific', 'East Asia & Pacific (IDA & IBRD countries)', 'East Asia & Pacific (excluding high income)',
        'Egypt, Arab Rep.', 'Euro area', 'Europe & Central Asia', 'Europe & Central Asia (IDA & IBRD countries)',
        'Europe & Central Asia (excluding high income)', 'European Union', 'Faroe Islands',
        'Fragile and conflict affected situations', 'French Polynesia', 'Gambia, The', 'Gibraltar',
        'Greenland', 'Guam', 'Heavily indebted poor countries (HIPC)', 'High income', 'Hong Kong SAR, China',
        'IBRD only', 'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total', 'Iran, Islamic Rep.',
        'Isle of Man', 'Korea, Dem. People\'s Rep.', 'Korea, Rep.', 'Kyrgyz Republic', 'Lao PDR',
        'Late-demographic dividend', 'Latin America & Caribbean', 'Latin America & Caribbean (excluding high income)',
        'Latin America & the Caribbean (IDA & IBRD countries)', 'Least developed countries: UN classification',
        'Low & middle income', 'Low income', 'Lower middle income', 'Macao SAR, China', 'Micronesia, Fed. Sts.',
        'Middle East & North Africa', 'Middle East & North Africa (IDA & IBRD countries)',
        'Middle East & North Africa (excluding high income)', 'Middle income', 'New Caledonia', 'North America',
        'Northern Mariana Islands', 'OECD members', 'Other small states', 'Pacific island small states',
        'Post-demographic dividend', 'Pre-demographic dividend', 'Puerto Rico', 'Russian Federation',
        'Sint Maarten (Dutch part)', 'Slovak Republic', 'Small states', 'South Asia', 'South Asia (IDA & IBRD)',
        'St. Kitts and Nevis', 'St. Lucia', 'St. Vincent and the Grenadines', 'Sub-Saharan Africa',
        'Sub-Saharan Africa (IDA & IBRD countries)', 'Sub-Saharan Africa (excluding high income)',
        'Syrian Arab Republic', 'Turkiye', 'Turks and Caicos Islands', 'Upper middle income',
        'Venezuela, RB', 'Virgin Islands (U.S.)'
    ]
    
    # Merge datasets on country information and year
    # Left join to keep all records from all.csv
    merged_df = all_df.merge(
        happiness_df[['Entity', 'Code', 'Year', 'Cantril ladder score']], 
        left_on=['Country Name', 'Country Code', 'Time'], 
        right_on=['Entity', 'Code', 'Year'], 
        how='left'
    )
    
    # Drop the duplicate Entity, Code and Year columns from the merge
    merged_df = merged_df.drop(['Entity', 'Code', 'Year'], axis=1)
    
    # Filter out data from year 2024
    merged_df = merged_df[merged_df['Time'] != 2024]
    
    # Filter out data for countries in the ignorance list
    merged_df = merged_df[~merged_df['Country Name'].isin(ignorance_country_list)]
    
    column_mapping = {
        # Basic identifiers
        'Country Name': 'Country',
        'Country Code': 'Code',
        'Time': 'Year',
        'Time Code': 'Year_Code',
        
        # Life expectancy columns
        'Life expectancy at birth, total (years) [SP.DYN.LE00.IN]': 'Life Expectancy',
        'Life expectancy at birth, male (years) [SP.DYN.LE00.MA.IN]': 'Life Expectancy Male',
        'Life expectancy at birth, female (years) [SP.DYN.LE00.FE.IN]': 'Life Expectancy Female',
        
        # Economic indicators
        'GDP (current US$) [NY.GDP.MKTP.CD]': 'GDP',
        'Gross national expenditure (% of GDP) [NE.DAB.TOTL.ZS]': 'National Expenditure',
        'Current account balance (% of GDP) [BN.CAB.XOKA.GD.ZS]': 'Current Account Balance',
        
        # Health expenditure
        'Domestic general government health expenditure (% of GDP) [SH.XPD.GHED.GD.ZS]': 'Government Health Expenditure',
        'Current health expenditure (% of GDP) [SH.XPD.CHEX.GD.ZS]': 'Total Health Expenditure',
        
        # Environmental indicators
        'Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e) [EN.GHG.CO2.MT.CE.AR5]': 'CO2 Emissions Total',
        'Carbon dioxide (CO2) emissions from Agriculture (Mt CO2e) [EN.GHG.CO2.AG.MT.CE.AR5]': 'CO2 Emissions Agriculture',
        'Carbon dioxide (CO2) emissions from Fugitive Emissions (Energy) (Mt CO2e) [EN.GHG.CO2.FE.MT.CE.AR5]': 'CO2 Emissions Energy',
        'Carbon dioxide (CO2) emissions from Industrial Processes (Mt CO2e) [EN.GHG.CO2.IP.MT.CE.AR5]': 'CO2 Emissions Industrial',
        'Carbon dioxide (CO2) emissions from Transport (Energy) (Mt CO2e) [EN.GHG.CO2.TR.MT.CE.AR5]': 'CO2 Emissions Transport',
        
        # Population indicators
        'Population, total [SP.POP.TOTL]': 'Population Total',
        'Population, male [SP.POP.TOTL.MA.IN]': 'Population Male',
        'Population, female [SP.POP.TOTL.FE.IN]': 'Population Female',
        'Population ages 0-14, female [SP.POP.0014.FE.IN]': 'Population 0-14 Female',
        'Population ages 0-14, male [SP.POP.0014.MA.IN]': 'Population 0-14 Male',
        'Population ages 0-14, total [SP.POP.0014.TO]': 'Population 0-14 Total',
        'Population ages 15-64, female [SP.POP.1564.FE.IN]': 'Population 15-64 Female',
        'Population ages 15-64, male [SP.POP.1564.MA.IN]': 'Population 15-64 Male',
        'Population ages 15-64, total [SP.POP.1564.TO]': 'Population 15-64 Total',
        'Population ages 65 and above, female [SP.POP.65UP.FE.IN]': 'Population 65+ Female',
        'Population ages 65 and above, male [SP.POP.65UP.MA.IN]': 'Population 65+ Male',
        'Population ages 65 and above, total [SP.POP.65UP.TO]': 'Population 65+ Total',
        
        # Health and mortality indicators
        'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total) [SH.DTH.COMM.ZS]': 'Deaths Communicable Diseases',
        'Cause of death, by non-communicable diseases (% of total) [SH.DTH.NCOM.ZS]': 'Deaths Non-Communicable',
        'Fertility rate, total (births per woman) [SP.DYN.TFRT.IN]': 'Fertility Rate',
        'Low-birthweight babies (% of births) [SH.STA.BRTW.ZS]': 'Low Birthweight',
        'Prevalence of anemia among non-pregnant women (% of women ages 15-49) [SH.ANM.NPRG.ZS]': 'Anemia Non-Pregnant Women',
        'Prevalence of anemia among women of reproductive age (% of women ages 15-49) [SH.ANM.ALLW.ZS]': 'Anemia Reproductive Age',
        
        # Nutrition and food security
        'Consumption of iodized salt (% of households) [SN.ITK.SALT.ZS]': 'Iodized Salt Consumption',
        'Food production index (2014-2016 = 100) [AG.PRD.FOOD.XD]': 'Food Production Index',
        
        # Urban and housing indicators
        'Population in largest city [EN.URB.LCTY]': 'Population Largest City',
        'Population living in slums (% of urban population) [EN.POP.SLUM.UR.ZS]': 'Population Slums',
        'Urban population [SP.URB.TOTL]': 'Urban Population',
        'Rural population [SP.RUR.TOTL]': 'Rural Population',
        'Population density (people per sq. km of land area) [EN.POP.DNST]': 'Population Density',
        
        # Trade indicators
        'Binding coverage, all products (%) [TM.TAX.MRCH.BC.ZS]': 'Trade Binding Coverage',
        
        # Happiness indicators
        'Cantril ladder score': 'Happiness Score'
    }

    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in merged_df.columns:
            merged_df = merged_df.rename(columns={old_name: new_name})

    # Modify data types
    # Ensure numeric data for key columns
    exclude_columns = ['Country', 'Year', 'Region', 'Income Group', 'Code', 'Year_Code']
    for col in merged_df.columns:
        if col not in exclude_columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Calculate GDP per capita after numeric conversion
    if 'GDP' in merged_df.columns and 'Population Total' in merged_df.columns:
        merged_df['GDP per Capita'] = merged_df['GDP'] / merged_df['Population Total']

    # Define data columns for later use
    data_columns = [col for col in merged_df.columns if col not in exclude_columns]

    # Fill missing Happiness Score values for 2013 with average of 2012 and 2014
    if 'Happiness Score' in merged_df.columns:
        for country in merged_df['Country'].unique():
            country_data = merged_df[merged_df['Country'] == country]
            
            # Get happiness scores for 2012 and 2014
            score_2012 = country_data[country_data['Year'] == 2012]['Happiness Score'].values
            score_2014 = country_data[country_data['Year'] == 2014]['Happiness Score'].values
            
            # If both 2012 and 2014 have values, calculate average for 2013
            if len(score_2012) > 0 and len(score_2014) > 0 and not pd.isna(score_2012[0]) and not pd.isna(score_2014[0]):
                avg_score = (score_2012[0] + score_2014[0]) / 2
                
                # Fill missing 2013 value
                mask = (merged_df['Country'] == country) & (merged_df['Year'] == 2013) & (pd.isna(merged_df['Happiness Score']))
                merged_df.loc[mask, 'Happiness Score'] = avg_score

    # Fill missing values for Israel in years 1960-1965 with column means
    israel_years = [1960, 1961, 1962, 1963, 1964, 1965]
    for year in israel_years:
        israel_mask = (merged_df['Country'] == 'Israel') & (merged_df['Year'] == year)
        if israel_mask.any():
            for col in data_columns:
                if col in merged_df.columns:
                    # Check if the value is missing for Israel in this year
                    if pd.isna(merged_df.loc[israel_mask, col]).any():
                        # Calculate mean of the column (excluding NaN values)
                        col_mean = merged_df[col].mean()
                        # Fill missing value with mean
                        merged_df.loc[israel_mask & pd.isna(merged_df[col]), col] = col_mean

    # Fill missing Fertility Rate values for Luxembourg with column mean
    if 'Fertility Rate' in merged_df.columns:
        luxembourg_mask = (merged_df['Country'] == 'Luxembourg') & (pd.isna(merged_df['Fertility Rate']))
        if luxembourg_mask.any():
            fertility_rate_mean = merged_df['Fertility Rate'].mean()
            merged_df.loc[luxembourg_mask, 'Fertility Rate'] = fertility_rate_mean

    # Drop rows where all data values are missing (excluding identifier columns)
    merged_df = merged_df.dropna(subset=data_columns, how='all')
    
    # Fill missing values with 0 for specific columns
    co2_zero_columns = ['CO2 Emissions Total', 'CO2 Emissions Energy', 'CO2 Emissions Transport', 'CO2 Emissions Industrial', 'CO2 Emissions Agriculture', 'Total Health Expenditure', 'Government Health Expenditure', 'Population Largest City', 'GDP per Capita', 'GDP', 'Happiness Score', 'Population Density']
    for col in co2_zero_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(0)

    # Divide economic and population indicators by 1 million for better readability
    # economic_population_columns = [
    #     'GDP per Capita', 'Population Total', 'Population Male', 'Population Female',
    #     'Population 0-14 Female', 'Population 0-14 Male', 'Population 0-14 Total',
    #     'Population 15-64 Female', 'Population 15-64 Male', 'Population 15-64 Total',
    #     'Population 65+ Female', 'Population 65+ Male', 'Population 65+ Total',
    #     'Population Largest City', 'Urban Population', 'Rural Population'
    # ]

    # for col in economic_population_columns:
    #     if col in merged_df.columns:
    #         merged_df[col] = merged_df[col] / 1_000_000

    # Drop specific columns that are not needed
    columns_to_drop = ['Deaths Non-Communicable', 'Deaths Communicable Diseases', 'Iodized Salt Consumption', 'Low Birthweight', 'Food Production Index',
                      'Anemia Non-Pregnant Women', 'Anemia Reproductive Age', 'Population Slums', 'Trade Binding Coverage', 'Current Account Balance', 'National Expenditure',
                      'Population 0-14 Female', 'Population 0-14 Male', 'Population 0-14 Total',
                      'Population 15-64 Female', 'Population 15-64 Male', 'Population 15-64 Total',
                      'Population 65+ Female', 'Population 65+ Male', 'Population 65+ Total']
    merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

    # Save the updated dataset
    merged_df.to_csv('data/all_with_happiness.csv', index=False)
    
    print(f"Dataset merged successfully!")
    print(f"Original dataset shape: {all_df.shape}")
    print(f"Merged dataset shape: {merged_df.shape}")
    # print(f"Records with Cantril ladder score: {merged_df['Cantril ladder score'].notna().sum()}")

if __name__ == "__main__":
    merge_datasets()
