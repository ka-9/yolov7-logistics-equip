import os
import pandas as pd

def create_csv(data_dir, phase):
    # Directory paths
    image_dir = os.path.join(data_dir, "images", phase)
    label_dir = os.path.join(data_dir, "labels", phase)

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
    csv_path = os.path.join(data_dir, f"{phase}.csv")
    df.to_csv(csv_path, index=False)

    print(f"CSV file saved successfully at {csv_path}")

if __name__ == "__main__":
    data_dir = "../BMW_dataset"
    # create_csv(data_dir, "train")
    create_csv(data_dir, "test")
