# importing all other files and libraries 
from Data_loading import loader
from Data_cleaning import encode_features, scale_features
from Visualizations import plot_heatmap, scatter_plot
from Models import train_linear_regression
from Evaluations import evaluate_model

# Load data
df = loader("medical_costs.csv")

# Preprocess
df = encode_features(df)
df = scale_features(df, ['Age', 'BMI', 'Children'])

# Visualize
plot_heatmap(df)
scatter_plot(df, 'BMI')
scatter_plot(df, 'Bge')

# Prepare data for modeling
X = df.drop(columns='Medical Cost')
y = df['Medical Cost']

# Train and evaluate model
model, X_test, y_test = train_linear_regression(X, y)
evaluate_model(model, X_test, y_test)
