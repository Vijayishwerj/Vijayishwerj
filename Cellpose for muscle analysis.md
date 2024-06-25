import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
from skimage.measure import regionprops
import pandas as pd

io.logger_setup()

# Initialize the model
model = models.Cellpose(gpu=False, model_type='cyto3')

# Path to your file
files = ['/Volumes/Extreme Pro/MCK characterization/ANALYSISTRAINING.tif']
imgs = [imread(f) for f in files]

# Define channels [cytoplasm, nucleus]
channels = [[2,1]]  # Adjust as needed based on your image type

# Perform segmentation
masks, flows, styles, diams = model.eval(imgs, diameter=None, channels=channels)

# Analyze each segmented cell
props = regionprops(masks[0], intensity_image=imgs[0])

# Collecting data
cell_data = {
    'Cell ID': [],
    'Area': [],
    'Centroid X': [],
    'Centroid Y': [],
    'Mean Intensity': []
}

for idx, prop in enumerate(props):
    cell_data['Cell ID'].append(idx + 1)
    cell_data['Area'].append(prop.area)
    cell_data['Centroid X'].append(prop.centroid[1])
    cell_data['Centroid Y'].append(prop.centroid[0])
    cell_data['Mean Intensity'].append(prop.mean_intensity)

# Convert to DataFrame
df = pd.DataFrame(cell_data)
print(df)

# Save results to a CSV file
df.to_csv('cell_analysis_results.csv', index=False)

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(imgs[0], cmap='gray')
ax.imshow(masks[0], alpha=0.5, cmap='jet')  # Overlay masks
plt.title('Segmented Cells')
plt.show()
