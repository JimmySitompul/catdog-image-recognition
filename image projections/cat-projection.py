import numpy as np
import cv2  # OpenCV library for image processing
from scipy.io import loadmat  # Library for reading MATLAB files
from scipy.linalg import svd
import pandas as pd  # Library for data manipulation and analysis

# Load the MATLAB file containing cat and dog images
mat = loadmat('C:/Users/Jimmy Sitompul/Downloads/catdog.mat')

# Print the contents of the MATLAB file
print("Matlab File")
print(mat)

# Extract the data from the MATLAB file and convert it to 
# a Pandas DataFrame
mat1 = pd.DataFrame(mat.get('T'))

# Convert the data to a NumPy array of float32 data type and flatten it
mat2 = np.array(mat1, dtype=np.float32).ravel()

# Print the length of the flattened NumPy array
print("Length of Array")
print(len(mat2))

# Extract the first 99 images (representing cats) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
cat = np.array(mat2[0:405504], dtype=np.float32)
cat = np.reshape(cat, (99, -1))
for i in range (0,99):
    img = cat[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    cat[i]=img

cat_pic = cat[0]

# Extract the next 99 images (representing dogs) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
dog = np.array(mat2[405504:811008], dtype=np.float32)
dog = np.reshape(dog, (99, -1))
for i in range (0,99):
    img = dog[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    dog[i]=img
    
dog = dog.T

# Print the cat and dog arrays and their lengths
print('Cat Pic Array')
print(cat_pic)
print('Dog Array')
print(dog)

meow = np.reshape(cat_pic,(len(cat_pic),-1))
print(meow.shape)

U,S,V_T = svd(dog, full_matrices=False)

print('Left Singular')
print(U)
print('Eigenvalues (diagonal)')
print(S)

k = 10
print(U[:,0:k].shape)

eigen = U[:,0]
eigen = np.array(eigen,dtype=np.float32).ravel()
print(eigen)

xmin= min(eigen)
xmax = max(eigen)

print('X minimum')
print(xmin)
print('X maximum')
print(xmax)

eigendogs=np.array([])

for i in eigen:
    f = lambda i : 21713.37796*i+480.94155
    val = int(f(i))
    eigendogs = np.append(eigendogs,val)

print("Eigendogs")
print(eigendogs)

eigendogs = np.reshape(eigendogs,(64,64)).astype(np.uint8)
cv2.imshow('Eigendog Image', eigendogs)

projection_cat = np.dot(U[:,0:k],np.dot(U[:,0:k].T,meow))
error_cat = meow - projection_cat

meow_pic = np.reshape(meow, (64, 64)).astype(np.uint8)
projection_cat_pic = np.reshape(projection_cat, (64, 64)).astype(np.uint8)
error_cat_pic = np.reshape(error_cat, (64, 64)).astype(np.uint8)

cv2.imshow('Cat Image', meow_pic)
cv2.imshow('Projected Cat Image', projection_cat_pic)
cv2.imshow('Error Cat Image', error_cat_pic)

cv2.waitKey(0)
cv2.destroyAllWindows()