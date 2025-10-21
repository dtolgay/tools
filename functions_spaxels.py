
import re
import pandas as pd 
import numpy as np

def remove_units(column_name:str):
    # Function to remove units from the column names
    return re.sub(r'\s*\[.*\]', '', column_name)


def remove_units_from_the_column_names_and_return_units(data:pd.DataFrame):
    
    # Initialize an empty dictionary to store the result
    units = {}

    # Loop through each column in the DataFrame
    for col in data.columns:
        # Extract the unit using regex
        unit = re.search(r'\[(.*?)\]', col)
        # Get the column name without the unit
        key = re.sub(r'\s*\[.*\]', '', col)
        # If a unit is found, add it to the dictionary; otherwise, set it as None
        units[key] = {"unit": unit.group(1) if unit else None}    
    
    data.columns = [remove_units(col) for col in data.columns] 
    
    return data, units