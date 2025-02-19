from utils import load_dataset, compute_baseline

# Load data
data_dir = "data/"
output_csv_path = f"{data_dir}/submissions/baseline.csv"

data_test = load_dataset(f"{data_dir}/test-images")
labels = compute_baseline(data_test)

new_columns = [f"Pixel {i}" for i in range(labels.shape[1])]
labels = labels.set_axis(new_columns, axis='columns')

new_index = [f"{i}.png" for i in range(labels.shape[0])]
labels = labels.set_axis(new_index, axis='index')

labels.T.to_csv(output_csv_path)