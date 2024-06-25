import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from cellpose import models, io
from skimage.io import imread

# Load image and model
model = models.Cellpose(gpu=True, model_type='cyto')
image_path = 'path_to_your_image.tif'
image = imread(image_path)
channels = [0, 0]  # Adjust depending on your image type

# Perform segmentation
masks, _, _, _ = model.eval([image], diameter=None, channels=channels)

# Create a figure and axis to display the image
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
mask_display = ax.imshow(masks[0], alpha=0.5, cmap='jet')  # Initial mask display

selected = np.ones(masks[0].max(), dtype=bool)  # State array for cell selections

def onclick(event):
    if event.inaxes == ax:
        x, y = int(event.xdata), int(event.ydata)
        cell_number = masks[0][y, x]
        if cell_number > 0:
            selected[cell_number - 1] = not selected[cell_number - 1]
            # Update the display
            mask_display.set_data(np.where(selected[masks[0]-1], masks[0], 0))
            fig.canvas.draw()

# Connect the click event to the onclick function
fig.canvas.mpl_connect('button_press_event', onclick)

# Add a button to save the refined mask
class Index:
    def save(self, event):
        refined_mask = np.where(selected[masks[0]-1], masks[0], 0)
        np.save('refined_mask.npy', refined_mask)  # Save refined mask as NumPy array
        print("Refined mask saved.")

callback = Index()
ax_save = plt.axes([0.81, 0.05, 0.1, 0.075])
b_save = Button(ax_save, 'Save Mask')
b_save.on_clicked(callback.save)

plt.show()
