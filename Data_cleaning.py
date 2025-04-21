import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
# encode specific collumn labels to make data processing easier
def encode_features(df): # takes the pandas dataframe as input
    le = LabelEncoder
    # will add labels later
    # format: 
    df['label'] = le.fit_transform(df['label'])
    return df

# scaling data for easier processing 
def scale_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df