import numpy as np
import cv2  # OpenCV library for image processing
from scipy.io import loadmat  # Library for reading MATLAB files
import pandas as pd  # Library for data manipulation and analysis


# Load the MATLAB file containing cat and dog images
mat = loadmat('C:/Users/Jimmy Sitompul/Downloads/catdog.mat')

# Print the contents of the MATLAB file
print(mat)

# Extract the data from the MATLAB file and convert it to 
# a Pandas DataFrame
mat1 = pd.DataFrame(mat.get('T'))

# Convert the data to a NumPy array of float32 data type and flatten it
mat2 = np.array(mat1, dtype=np.float32).ravel()

# Print the length of the flattened NumPy array
print(len(mat2))

# Extract the first 99 images (representing cats) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
cat = np.array(mat2[0:405504], dtype=np.float32)
cat = np.reshape(cat, (99, 4096))
for i in range (0,99):
    img = cat[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    cat[i]=img

# Extract the next 99 images (representing dogs) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
dog = np.array(mat2[405504:811008], dtype=np.float32)
dog = np.reshape(dog, (99, 4096))
for i in range (0,99):
    img = dog[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    dog[i]=img

# Print the cat and dog arrays and their lengths
print(len(cat))
print(len(dog))

# Calculate the average of the cat and dog arrays along 
# the columns (i.e., across the images)
avg_cat = np.mean(cat, axis=0)
print(avg_cat)
avg_dog = np.mean(dog, axis=0)
print(avg_dog)

# Reshape the average cat and dog arrays to 2D arrays of 64x64 size 
# and convert them to unsigned integer data type
avg_image_cat = np.reshape(avg_cat, (64, 64)).astype(np.uint8)
avg_image_dog = np.reshape(avg_dog, (64, 64)).astype(np.uint8)

# Display the average cat and dog images using OpenCV's imshow function
cv2.imshow("Cat Average Face", avg_image_cat)
cv2.imshow("Dog Average Face", avg_image_dog)

# Wait for a key press and destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()