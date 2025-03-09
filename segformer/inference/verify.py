import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the CSV file. Assuming the first column is the row index.
    csv_file = "submission.csv"
    try:
        df = pd.read_csv(csv_file, index_col=0)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
        return
    
    # Shape of the CSV
    shape = df.shape
    print("Shape of CSV:", shape)
    
    # Unique values in the entire table
    unique_values = np.unique(df.values)
    print("Unique values in the entire table:", unique_values)
    
    # Flatten the dataframe values and compute counts
    flat_values = df.values.flatten()
    counts = pd.Series(flat_values).value_counts().sort_index()
    
    print("\nCounts for each class:")
    for cls, count in counts.items():
        print(f"Class {cls}: {count}")
    
    # Percentage of non-zero classes (non-zero pixels)
    total_pixels = df.size
    non_zero_count = (df != 0).sum().sum()
    percentage_non_zero = (non_zero_count / total_pixels) * 100
    print(f"\nPercentage of non-zero classes: {percentage_non_zero:.2f}%")
    
    # Exclude class 0 from the plot
    counts = counts[counts.index != 0]
    
    # Plot frequency of classes
    plt.figure(figsize=(10, 6))
    counts.plot(kind='bar')
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Frequency of Classes (excluding class 0)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot as "plot.png"
    plot_filename = "plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.show()

if __name__ == "__main__":
    main()
