from sklearn.preprocessing import LabelEncoder, StandardScaler
# encode specific collumn labels to make data processing easier


def encode_features(df): # takes the pandas dataframe as input
    le = LabelEncoder
    # will add labels later
    # format: 
    df['Age'] = le.fit_transform(df['Age'])
    df['Sex'] = le.fit_transform(df['Sex'])
    df['BMI'] = le.fit_transform(df['BMI'])
    df['Children'] = le.fit_transform(df['Children'])
    df['Smoker'] = le.fit_transform(df['Smoker'])
    df['Region'] = le.fit_transform(df['Region'])
    # reminder to do sommething for the whole target collumn thing
    return df

# scaling data for easier processing 
def scale_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df