import pandas as pd

def rename_cols(df):
    """
    Renames the DataFrame columns to improve clarity and consistency.
    In this case, the function performs two renaming operations:

    1. Renames specific columns, such as 'Unnamed: 11' to 'fatal' and 'Species ' to 'species'. 
       The 'fatal' column will be our reference point for identifying fatal injuries.
    
    2. Converts all column names to lowercase and replaces spaces with underscores.
       This step standardizes the column names, making it easier to work with them.

    Parameters:
        df (pandas.DataFrame): The DataFrame whose columns will be renamed.

    Returns:
        pandas.DataFrame: The DataFrame with renamed columns.
    """
    
    # Rename specific columns for clarity
    df.rename(columns={"Unnamed: 11": "fatal", "Species ": "species"}, inplace=True)
    
    # Convert the rest of the column names to lowercase and replace spaces with underscores.
    df.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
    
    return df


def remove_nulls(df):
    """
    Removes rows with missing values in key columns for our analysis and returns the modified DataFrame.
    This step eliminates rows where data is missing in essential columns such as 'country', 'name', 'sex', 'age',
    and 'fatal'.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The modified DataFrame, without rows that have missing values in key columns.
    """
    df = df.dropna(subset=['country','name', 'sex', 'age', 'fatal'])
    return df


def change_float_to_int(df):
    """
    Converts float64 columns to their integer equivalents.
    Since some numeric columns, such as 'age' or 'year', are stored as floats due to the presence of null values or 
    decimals, this function converts them to integers as decimals are not relevant. Null values are handled by assigning 
    them 0 to prevent errors in later calculations.

    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.

    Returns:
        pandas.DataFrame: The modified DataFrame with float64 columns converted to integers.
    """
    df = df.apply(lambda x: x.fillna(0).astype(int) if x.dtype == 'float64' else x)
    return df


def remove_small_reps(df):
    """
    Removes small representations from each column, keeping only those that have at least 30 occurrences.
    This function reviews all columns and focuses on string-type columns.
    
    First, it removes unnecessary leading and trailing spaces from strings. Then, 
    it filters rows in each column, retaining only those that appear at least 30 times in the column.
    This is useful for removing low-representation values that might skew the analysis.

    Parameters:
        df (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The DataFrame with small representations removed and text formatting corrected.
    """
    for x in df.columns:
        if df[x].dtype == "str":  # If the column is of string type, remove leading and trailing spaces.
            df[x] = df[x].str.strip()
        # Filter the column to keep only values that appear at least 30 times.
        df[x] = df[x].loc[df[x].isin(df[x].value_counts()[lambda x: x >= 30].index)]
    return df


def clean_str_punctuation(df):
    """
    Cleans by removing punctuation, adjusting white spaces, and applying title case to text strings. 
    This function is used in columns such as 'country', 'state', and 'location'.

    First, a translation table is created to remove common punctuation characters like commas, periods, 
    exclamation marks, and question marks. Then, the function iterates over each column in the DataFrame, 
    and if the column is of string type, it performs the transformations.

    Parameters:
        df (pandas.DataFrame): The DataFrame to clean.

    Returns:
        pandas.DataFrame: The DataFrame with formatted text strings and no punctuation.
    """
    # Define the punctuation characters to be removed.
    mytable = str.maketrans('', '', '¡¿.,!?;')
    
    # Iterate through each column and apply the cleaning only to 'object' type (text) columns.
    for x in df.columns:
        if df[x].dtype == "object":  # If the column is of string type.
            df[x] = df[x].str.strip().str.title().str.translate(mytable)
    
    # Create a subset of string-type columns and apply the same cleaning.
    df_clean = df.select_dtypes(include=['object'])
    df_clean = df_clean.apply(lambda x: x.str.strip().str.title().str.translate(mytable))
    
    # Replace the original columns with the cleaned ones.
    df = df.drop(df_clean.columns, axis=1).join(df_clean)
    
    return df


def clean_age_column(df):
    """
    Cleans and standardizes the 'age' column, ensuring that numeric values are handled correctly.

    The 'age' column may contain values in different formats, including additional text or descriptions. 
    This function extracts only the numeric part and converts the values to a numeric format (float).
    
    - Splits the string by spaces and takes the first value (which is usually the age) before any additional text.
    - Uses `pd.to_numeric` to convert the age to a numeric format. Any value that cannot be converted 
      (e.g., text) is turned into `NaN`.
    - Non-convertible or missing values are kept as `NaN` to ensure that the column is suitable for numeric analysis.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'age' column to clean.

    Returns:
        pandas.DataFrame: The DataFrame with the 'age' column cleaned and in numeric format.
    """
    import numpy as np

    # Split the string and take the first part (the age), then convert to numeric.
    df["age"] = df["age"].str.split(" ").str[0].apply(pd.to_numeric, errors="coerce")
    
    # Fill non-convertible values with NaN.
    df["age"].fillna(value=np.nan, inplace=True)
    
    return df


def clean_fatal_column(df):
    """
    Cleans and standardizes the 'fatal' column.
    This function performs the following transformations:
    
    1. Removes white spaces and applies uppercase formatting: Removes unnecessary white spaces 
       and converts all text to uppercase.
    2. Standardizes the different ways fatality is recorded, converting:
       - 'Y', 'F', and 'Y X 2' to 'Yes'.
       - 'N', 'N N' to 'No'.
       - Ambiguous values such as 'M', 'NQ', and 'UNKNOWN' to 'Unknown'.
    3. Missing or null values in the column are replaced with 'Unknown'.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'fatal' column to clean.

    Returns:
        pandas.DataFrame: The DataFrame with the 'fatal' column cleaned and standardized.
    """
    
    # Clean white spaces, convert to uppercase, and replace specific values
    df["fatal"] = df["fatal"].str.strip().str.upper().replace({
        'Y': 'Yes',  # Convert 'Y' to 'Yes'
        'N': 'No',  # Convert 'N' to 'No'
        'F': 'Yes',  # Convert 'F' to 'Yes' (presumably indicates fatality)
        'N N': 'No',  # Correct variations of 'No'
        'UNKNOWN': 'Unknown',  # Standardize 'Unknown'
        'M': 'Unknown',  # Assume 'M' means 'Unknown'
        'NQ': 'Unknown',  # Assume 'NQ' means 'Unknown'
        'Y X 2': 'Yes'  # Assume 'Y X 2' means 'Yes'
    })
    
    # Fill missing values with 'Unknown'
    df['fatal'] = df['fatal'].fillna('Unknown')
    
    return df


def standardize_time(time_str):
    """
    Standardizes a time string to a 24-hour format.

    Time data can come in a variety of formats, such as 'dawn', 'afternoon', or '6:30 PM'.
    This function converts all possible formats to a uniform representation in a 24-hour format, 
    making it easier to analyze the times of the attacks. If the time is not available, 
    a default value of '12:00' is used.

    Parameters:
        time_str (str or int): The string representing the time to standardize.

    Returns:
        str: The standardized time string in a 24-hour format.
    """
    if pd.isna(time_str):
        return '12:00'
    if isinstance(time_str, int):
        time_str = str(time_str)
    time_str = time_str.strip().lower()
    
    # Handle different descriptions of time
    if 'early' in time_str or 'dawn' in time_str or 'before' in time_str:
        return '06:00'
    if 'morning' in time_str:
        return '09:00'
    if 'midday' in time_str or 'noon' in time_str:
        return '12:00'
    if 'afternoon' in time_str:
        return '15:00'
    if 'evening' in time_str or 'dusk' in time_str or 'sunset' in time_str:
        return '18:00'
    if 'night' in time_str or 'midnight' in time_str:
        return '23:00'
    
    # Attempt to parse different time formats
    try:
        time_str = time_str.replace('h', ':').replace(' ', '')
        if '-' in time_str:
            time_str = time_str.split('-')[0].strip()
        if ':' in time_str:
            return pd.to_datetime(time_str, format='%H:%M', errors='coerce').strftime('%H:%M')
        if ' ' in time_str:
            time_str = time_str.split()[0]
        time_str = time_str.replace('j', '').replace('"', '').replace('pm', '').replace('am', '')
        if len(time_str) == 4:
            return f'{time_str[:2]}:{time_str[2:]}'
        if len(time_str) == 3:
            return f'0{time_str[0]}:{time_str[1:]}'
        return '12:00'  # Default value if conversion fails
    except:
        return '12:00'
    

def clean_time_column(df):
    """
    Cleans and standardizes the 'time' column in the DataFrame by applying the 'standardize_time' 
    function to each value.

    The 'time' column may contain data in various formats (such as descriptive text or hours in 
    different formats). This function ensures that all time values are in a uniform format 
    (for example, in 24-hour format).
    
    It uses the 'standardize_time' function to convert values like 'morning', '6:30 PM', or 'dusk' 
    into a consistent time format.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'time' column to clean.

    Returns:
        pandas.DataFrame: The DataFrame with the 'time' column cleaned and standardized.
    """
    df["time"] = df["time"].apply(standardize_time)  # Applies time standardization
    return df


"""
Set of valid shark species for cleaning and normalizing the 'species' column.

This set includes the names of shark species that are considered valid and will be used to 
compare and normalize the data in the 'species' column. During the cleaning process, values 
that do not match one of these species will be replaced with 'Unknown' or assigned to the 
closest species name.
"""


def clean_species(species_str, valid_species):
    """
    Cleans and standardizes species names in the 'species' column by comparing with a list of 
    valid species.

    The function aims to normalize the value of the 'species' column by comparing each value with a 
    predefined list of valid species. If the value matches (case insensitive) one of the valid 
    species, the corresponding species name is returned. If no match is found, the value 'Unknown' 
    is assigned.

    - Strips any leading or trailing whitespace from the string.
    - If the species is in the list of valid species, the species name is returned.
    - If there is no match or the value is null, 'Unknown' is assigned.

    Parameters:
        species_str (str): The species name to clean.
        valid_species (set): A set of valid species for comparison.

    Returns:
        str: The cleaned species name, or 'Unknown' if no match is found.
    """
    if pd.isna(species_str):
        return 'Unknown'
    species_str = str(species_str).strip()
    # Extract only the main species name if it matches the valid ones
    for name in valid_species:
        if name.lower() in species_str.lower():
            return name
    return 'Unknown'  # Default value if no match is found


def clean_species_column(df, valid_species):
    """
    Applies the species cleaning using the 'clean_species' function to ensure all species 
    are valid and consistent.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'species' column.
        valid_species (set): Set of valid species for cleaning.

    Returns:
        pandas.DataFrame: The DataFrame with the 'species' column cleaned.
    """
    df['species'] = df['species'].apply(lambda x: clean_species(x, valid_species))
    return df


def clean_pdf(pdf_str):
    """
    Cleans a string representing a PDF file by removing any non-alphanumeric characters, 
    except for periods, underscores, and dashes.

    This function ensures that the PDF file names in the DataFrame are cleaned and standardized, 
    removing special characters that could cause issues. If the value is a number, it converts 
    it to a string. If the string is empty or invalid, 'Unknown' is assigned.

    Parameters:
        pdf_str (str or int): The value of the 'pdf' column to clean.

    Returns:
        str: The cleaned PDF file name or 'Unknown' if the value is invalid.
    """
    if pd.isna(pdf_str):
        return 'Unknown'
    if isinstance(pdf_str, int):
        pdf_str = str(pdf_str)
    pdf_str = str(pdf_str).strip()
    # Remove non-alphanumeric characters except periods, underscores, and dashes
    pdf_str = ''.join(c for c in pdf_str if c.isalnum() or c in ['.', '_', '-'])
    return pdf_str if pdf_str else 'Unknown'


def clean_pdf_column(df):
    """
    Applies the 'clean_pdf' function to the 'pdf' column to clean and standardize the PDF file names.

    Iterates through the 'pdf' column of the DataFrame, cleaning each value to ensure that the file names 
    are in a proper format for handling.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the 'pdf' column.

    Returns:
        pandas.DataFrame: The DataFrame with the 'pdf' column cleaned.
    """
    df['pdf'] = df['pdf'].apply(clean_pdf)
    return df


def drop_useless_columns(df):
    """
    Drops unnecessary or irrelevant columns from the DataFrame.

    In this case, the columns 'original_order', 'unnamed:_21', and 'unnamed:_22' are removed because 
    they do not provide relevant information for the analysis. This function ensures that the DataFrame 
    remains clean and contains only useful columns.

    Parameters:
        df (pandas.DataFrame): The DataFrame from which the columns will be dropped.

    Returns:
        pandas.DataFrame: The DataFrame without the unnecessary columns.
    """
    df = df.drop(["original_order", "unnamed:_21", "unnamed:_22"], axis=1)
    return df


def main_cleaning(df_main, valid_species):
    """
    Main cleaning function that runs a series of steps to prepare the data for analysis.

    This function applies a series of transformations to the main DataFrame to clean and standardize the data. 
    It performs the following tasks:
    
    - Renames columns for better readability.
    - Removes duplicates and missing values in key columns.
    - Converts numeric columns from floats to integers.
    - Removes categories with low representation.
    - Cleans punctuation in various text columns.
    - Normalizes and cleans values.
    - Drops unnecessary columns.

    Parameters:
        df_main (pandas.DataFrame): The main DataFrame to clean.
        valid_species (set): Set of valid species for the 'species' column.

    Returns:
        pandas.DataFrame: The cleaned DataFrame, ready for analysis.
    """
    df_main = rename_cols(df_main)  # Rename columns
    df_main = remove_nulls(df_main)  # Remove duplicates
    df_main = change_float_to_int(df_main)  # Convert floats to integers
    df_main = remove_small_reps(df_main)  # Remove small representations
    df_main = clean_str_punctuation(df_main)  # Clean punctuation in text
    df_main = clean_age_column(df_main)  # Clean the age column
    df_main = clean_fatal_column(df_main)  # Clean the 'fatal' column
    df_main = clean_time_column(df_main)  # Standardize the time column
    df_main = clean_species_column(df_main, valid_species)  # Clean the 'species' column
    df_main = clean_pdf_column(df_main)  # Clean the 'pdf' column
    df_main = drop_useless_columns(df_main)  # Drop unnecessary columns
    
    return df_main