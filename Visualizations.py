# creating a visual example for powerpoint & easy interpretation
import seaborn as sb
import matplotlib.pyplot as plt

# heatmap
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    sb.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

# scatterplot function with matplotlib 
def scatterplot(df, x_column, y_column='Medical Cost'):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.5)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Relationship between {x_column} and Medical Costs')
    plt.grid(True)
    plt.show()
