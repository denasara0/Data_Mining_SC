# Data_Mining_SC
This Repository contains the source code for group 20's Data mining project and a copy of the dataset &amp; LaTeX formatting code.
## Project Overview
While our file names are fairly straightforward, here is a comprehensive guide to each file in the repository. For organizational purposes, each function group was created in its own file.
```
pandas 
numpy 
matplotlib
seaborn
scikit-learn
pytest
```
### Dataset
The dataset used is from the Kaggle database, and is named "Medical_costs.csv"
Check out the dataset [here](https://www.kaggle.com/datasets/waqi786/medical-costs)

### Data Loading
We used the Pandas library to ease the task of preprocessing. Loading our CSV file into a pandas dataframe helped keep datatypes consistent, while also simplifying the cleaning function by taking out one extra step.
```
def loader(data) -> pd.DataFrame:
```
### Data Cleaning
This section of the program encodes names in the columns for our dataset, and declares the target column

### Models


