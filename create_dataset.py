import pandas as pd

def merge_datasets():
    # Load the datasets
    all_df = pd.read_csv('data/all.csv')
    happiness_df = pd.read_csv('data/happiness-cantril-ladder.csv')
    
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
        'GDP (current US$) [NY.GDP.MKTP.CD]': 'GDP per Capita',
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

    # Divide economic and population indicators by 1 million for better readability
    economic_population_columns = [
        'GDP per Capita', 'Population Total', 'Population Male', 'Population Female',
        'Population 0-14 Female', 'Population 0-14 Male', 'Population 0-14 Total',
        'Population 15-64 Female', 'Population 15-64 Male', 'Population 15-64 Total',
        'Population 65+ Female', 'Population 65+ Male', 'Population 65+ Total',
        'Population Largest City', 'Urban Population', 'Rural Population'
    ]

    for col in economic_population_columns:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col] / 1_000_000

    # Save the updated dataset
    merged_df.to_csv('data/all_with_happiness.csv', index=False)
    
    print(f"Dataset merged successfully!")
    print(f"Original dataset shape: {all_df.shape}")
    print(f"Merged dataset shape: {merged_df.shape}")
    # print(f"Records with Cantril ladder score: {merged_df['Cantril ladder score'].notna().sum()}")

if __name__ == "__main__":
    merge_datasets()
