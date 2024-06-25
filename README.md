import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from skimage.io import imread
from skimage.measure import regionprops

# Setup logger for cellpose
io.logger_setup()

# Initialize the Cellpose model, attempting to use GPU acceleration
model = models.Cellpose(gpu=True, model_type='cyto3')

# Define the path to your image
image_path = '/Volumes/Extreme Pro/MCK characterization/ANALYSISTRAINING.tif'
image = imread(image_path)

# Ensure the image has the expected number of channels
if image.ndim == 3 and image.shape[2] == 4:
    # Extract channel 3 (assuming 0-based index, so channel 3 is the fourth channel)
    channel_3_image = image[:, :, 3]
else:
    print("Image does not have the expected number of channels.")
    exit()

# Segment the image using Cellpose - specify this is a single-channel image
# Assuming the fluorescence signal of interest for segmentation is in this channel
masks, flows, styles, diams = model.eval([channel_3_image], diameter=None, channels=[3, 3])

# Calculating centroids using regionprops
props = regionprops(masks[0])

# Extract centroids and plot them
centroids = np.array([prop.centroid for prop in props])  # List of (y, x) coordinates

# Create a figure to show the image and centroids
fig, ax = plt.subplots()
ax.imshow(channel_3_image, cmap='gray')
ax.scatter(centroids[:, 1], centroids[:, 0], color='red', s=20)  # Note: centroids[:, 1] is x, centroids[:, 0] is y
plt.title('Centroids in Channel 3')
plt.show()
