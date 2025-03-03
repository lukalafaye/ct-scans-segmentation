import pandas as pd

# Read the entire CSV and select only 10 random rows
train_df = pd.read_csv("y_train.csv", index_col=0).sample(n=30, axis=1, random_state=42)

# Save the selected subset to a new CSV file
train_df.to_csv("y_train_sampled.csv", index=True)

print("Saved y_train_sampled.csv with shape:", train_df.shape)
