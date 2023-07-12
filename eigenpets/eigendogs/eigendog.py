import numpy as np
import cv2  # OpenCV library for image processing
from scipy.io import loadmat  # Library for reading MATLAB files
from scipy.linalg import svd
import pandas as pd  # Library for data manipulation and analysis
import numpy as np
from sklearn.preprocessing import normalize

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

# Extract the first 99 images (representing cats) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
cat = np.array(mat2[0:405504], dtype=np.float32) #99*4096 = 405504
cat = np.reshape(cat, (99, -1))
for i in range (0,99):
    img = cat[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    cat[i]=img



# Extract the next 99 images (representing dogs) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
dog = np.array(mat2[405504:811008], dtype=np.float32) #405504*2 = 811008
dog = np.reshape(dog, (99, -1))
for i in range (0,99):
    img = dog[i]
    img = np.reshape(img, (64, 64))
    img = np.rot90(img, k=3)
    img = np.array(img, dtype=np.float32).ravel()
    dog[i]=img
    
for img in dog:
    img = np.reshape(img, (64, 64))
    mirror_img = np.fliplr(img)
    mirror_img = np.array(mirror_img, dtype=np.float32).ravel()
    dog = np.append(dog,mirror_img)

dog = np.reshape(dog, (198, -1))
dog = dog.T
print(dog.shape)  

U,S,V_T = svd(dog, full_matrices=False)

k = 10

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
    f = lambda i : 23017.78299*i+504.26438
    val = int(f(i))
    eigendogs = np.append(eigendogs,val)

print("Eigendogs")
print(eigendogs)

eigendogs = np.reshape(eigendogs,(64,64)).astype(np.uint8)
cv2.imshow('Eigendog Image', eigendogs)

cv2.waitKey(0)
cv2.destroyAllWindows()