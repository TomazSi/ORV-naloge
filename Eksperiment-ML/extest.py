import h5py
import numpy as np
import matplotlib.pyplot as plt

# Funkcija za prikaz slike
def show_image(image_array):
    # Denormaliziramo sliko
    image_array = (image_array * 255).astype(np.uint8)
    plt.imshow(image_array)
    plt.axis('off')
    plt.show()

# Pot do vaše .h5 datoteke
h5_file_path = 'data.h5'

# Odpiranje .h5 datoteke
with h5py.File(h5_file_path, 'r') as h5_file:
    # Preberemo podatke
    X_train = h5_file['X_train'][:]
    y_train = h5_file['y_train'][:]
    X_val = h5_file['X_val'][:]
    y_val = h5_file['y_val'][:]
    X_test = h5_file['X_test'][:]
    y_test = h5_file['y_test'][:]

    # Prikaz nekaj slik iz učnega niza
    num_images_to_display = 5
    for i in range(num_images_to_display):
        print(f"Label: {y_train[i]}")
        show_image(X_train[i])
