import numpy as np
import cv2  # OpenCV library for image processing
from scipy.io import loadmat  # Library for reading MATLAB files
from scipy.linalg import svd
import pandas as pd  # Library for data manipulation and analysis
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

# Load the MATLAB file containing cat and dog images
mat = loadmat('C:/Users/Jimmy Sitompul/Downloads/catdog.mat')

# Print the contents of the MATLAB file
print(mat)
print(len(mat.get("T")))
print([len(a) for a in mat.get("T")])

# Extract the data from the MATLAB file and convert it to 
# a Pandas DataFrame
mat1 = pd.DataFrame(mat.get('T'))

# Convert the data to a NumPy array of float32 data type and flatten it
mat2 = np.array(mat1, dtype=np.float32).ravel()

# Print the length of the flattened NumPy array
print(len(mat2))
print(len(mat2)/2)

# Extract the next 99 images (representing dogs) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
cat = np.array(mat2[0:405504], dtype=np.float32) #405504*2 = 811008
cat = np.reshape(cat, (99, -1))
for i in range (0,99):
    img = cat[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    cat[i]=img
cat_pic = cat[0]

meow = np.reshape(cat_pic,(64,-1))
print(meow.shape)

# Construct the even and odd basis vectors
v_even = (meow + np.fliplr(meow))/2
v_odd = (meow - np.fliplr(meow))/2

# Project the image onto each of the basis vectors
c_even = np.sum(meow*v_even)/np.sum(v_even**2)
c_odd = np.sum(meow*v_odd)/np.sum(v_odd**2)



# Symmetrize the image by adding the even component to itself reflected about the vertical axis
# and subtracting the odd component from itself reflected about the vertical axis
sym_cat = c_even*v_even + c_even*np.fliplr(v_even) - c_odd*v_odd - c_odd*np.fliplr(v_odd)

sym_cat = np.array(sym_cat, dtype=np.float32).ravel()
xmin= min(sym_cat)
print(xmin)
xmax = max(sym_cat)
print(xmax)



symmetrized_cat = np.array([])
# Convert the symmetrized image to uint8 format
for i in sym_cat:
    f = lambda i : (85/124)*i+(-170/31)
    val = int(f(i))
    symmetrized_cat = np.append(symmetrized_cat,val)
symmetrized_cat = np.reshape(symmetrized_cat,(64,64)).astype(np.uint8)
print(symmetrized_cat.dtype)

meow= meow.astype(np.uint8)

cv2.imshow('Cat Image', meow)
cv2.imshow('Symmetrized Cat Image', symmetrized_cat)
cv2.waitKey(0)
cv2.destroyAllWindows()