import pandas as pd


# creating a data loading function
def loader(data) -> pd.DataFrame: # turning data into a pandas dataframe for easy use
    df =pd.read_csv("medical_costs.csv")
    return df

# adding more later