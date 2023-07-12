import numpy as np
import cv2  # OpenCV library for image processing
from scipy.io import loadmat  # Library for reading MATLAB files
from scipy.linalg import svd
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
cat = np.reshape(cat, (99, -1))
for i in range (0,99):
    img = cat[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    cat[i]=img
cat = cat.T

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
dog_pic = dog[0]

# Print the cat and dog arrays and their lengths
print('Dog Pic Array')
print(dog_pic)
print('Cat Array')
print(cat)

woof = np.reshape(dog_pic,(len(dog_pic),-1))

U,S,V_T = svd(cat, full_matrices=False)

print('Left Singular')
print(U)
print('Eigenvalues (diagonal)')
print(S)

k = 10

eigen=U[:,0]
print('Eigen')
print(eigen)

xmin= min(eigen)
xmax = max(eigen)

print('X minimum')
print(xmin)
print('X maximum')
print(xmax)

eigencats=np.array([])

for i in eigen:
    f= lambda i: 33109.26153*i+651.84401
    val = int(f(i))
    eigencats = np.append(eigencats,val)
    
print('Eigencats')
print(eigencats)

eigencats = np.reshape(eigencats,(64,64)).astype(np.uint8)
cv2.imshow('Eigencat Image', eigencats)

projection_dog = np.dot(U[:,0:k],np.dot(U[:,0:k].T,woof))
error_dog = woof - projection_dog

woof_pic = np.reshape(woof, (64, 64)).astype(np.uint8)
projection_dog_pic = np.reshape(projection_dog, (64, 64)).astype(np.uint8)
error_dog_pic = np.reshape(error_dog, (64, 64)).astype(np.uint8)

cv2.imshow('Dog Image', woof_pic)
cv2.imshow('Projected Dog Image', projection_dog_pic)
cv2.imshow('Error Dog Image', error_dog_pic)

cv2.waitKey(0)
cv2.destroyAllWindows()