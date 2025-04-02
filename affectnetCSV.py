import os
import csv
import numpy as np

# Path to your annotations and images
annotations_dir = "C:\\Users\\andre\\Documents\\train_set\\annotations"
images_dir = "C:\\Users\\andre\\Documents\\train_set\\images"
csv_output_path = "affectnet.csv"

# Common image extensions to try
image_extensions = [".jpg", ".png", ".jpeg"]

rows = []
for filename in os.listdir(annotations_dir):
    # We will only process the expression file, then grab the others
    # filename pattern example: "12345_exp.npy"
    if filename.endswith("_exp.npy"):
        # Strip off "_exp.npy" to get the base ID (e.g., "12345")
        base_id = filename[:-8]  # remove last 8 chars: '_exp.npy'
        
        # Build paths for the 4 annotation files
        exp_path = os.path.join(annotations_dir, f"{base_id}_exp.npy")
        val_path = os.path.join(annotations_dir, f"{base_id}_val.npy")
        aro_path = os.path.join(annotations_dir, f"{base_id}_aro.npy")
        lnd_path = os.path.join(annotations_dir, f"{base_id}_lnd.npy")
        
        # Check that all files exist (optional, but good practice)
        if not (os.path.isfile(exp_path) and 
                os.path.isfile(val_path) and 
                os.path.isfile(aro_path) and 
                os.path.isfile(lnd_path)):
            print(f"Skipping {base_id} because not all .npy files are present.")
            continue
        
        # Load each annotation
        expression = np.load(exp_path)  # often an integer label
        valence = np.load(val_path)     # float in [-1, 1]
        arousal = np.load(aro_path)     # float in [-1, 1]
        landmarks = np.load(lnd_path)   # array of shape (N, 2) for N landmarks
        
        # Locate the corresponding image file
        image_path = None
        for ext in image_extensions:
            candidate = os.path.join(images_dir, base_id + ext)
            if os.path.isfile(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            print(f"Warning: No image found for {base_id}. Skipping.")
            continue
        
        # Prepare a row for the CSV
        # landmarks is an array; you might store it as a string or JSON
        rows.append([
            image_path,
            int(expression) if expression.size == 1 else expression.tolist(),
            float(valence)   if valence.size == 1   else valence.tolist(),
            float(arousal)   if arousal.size == 1   else arousal.tolist(),
            landmarks.tolist()  # or str(landmarks.tolist())
        ])

# Write out to CSV
header = ["image_path", "expression", "valence", "arousal", "landmarks"]
with open(csv_output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print(f"CSV saved to {csv_output_path}. Rows written: {len(rows)}")
