# creating a visual example for powerpoint & easy interpretation
import seaborn as sb
import matplotlib.pyplot as plt

# placeholder functions for actual visualization
def plot_heatmap(df):
    plt.figure(figsize=(10, 8))
    sb.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def scatter_plot(df, x_col, y_col='charges'):
    sb.scatterplot(x=x_col, y=y_col, data=df)
    plt.title(f"{y_col} vs {x_col}")
    plt.show()