import os
import pandas as pd

# Directory containing images and labels
data_dir = "../BMW_dataset"

# Directory paths
image_dir = os.path.join(data_dir, "images", "train")
label_dir = os.path.join(data_dir, "labels", "train")

# Get list of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

# Create a list to store image file names and corresponding label file names
data = []

# Iterate over image files and get corresponding label file names
for image_file in image_files:
    label_file = os.path.splitext(image_file)[0] + ".txt"
    data.append((image_file, label_file))

# Convert data to DataFrame
df = pd.DataFrame(data, columns=["image_file", "label_file"])

# Save DataFrame to CSV
csv_path = os.path.join(data_dir, "train.csv")
df.to_csv(csv_path, index=False)

print(f"CSV file saved successfully at {csv_path}")
