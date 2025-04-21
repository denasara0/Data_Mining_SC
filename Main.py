# importing all other files and libraries 
from Data_loading import loader
from Data_cleaning import encode_features, scale_features
from Visualizations import plot_heatmap, scatter_plot
from Models import train_linear_regression
from Evaluations import evaluate_model

# Load data
df = loader("data/insurance.csv")

# Preprocess
df = encode_features(df)
df = scale_features(df, ['age', 'bmi', 'children'])

# Visualize
plot_heatmap(df)
scatter_plot(df, 'bmi')
scatter_plot(df, 'age')

# Prepare data for modeling
X = df.drop(columns='charges')
y = df['charges']

# Train and evaluate model
model, X_test, y_test = train_linear_regression(X, y)
evaluate_model(model, X_test, y_test)
