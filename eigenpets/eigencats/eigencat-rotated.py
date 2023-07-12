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
cat_mirror = np.fliplr(cat)
cat = np.array(cat, dtype=np.float32)
cat_mirror = np.array(cat_mirror, dtype=np.float32)

kitty_cat = np.append(cat,cat_mirror)
kitty_cat = np.reshape(kitty_cat, (198,-1))
kitty_cat = kitty_cat.T

# Extract the next 99 images (representing dogs) from the flattened array 
# and reshape it to a 2D array of length 99 whose elements are sub-arrays 
# of length 4096 
dog = np.array(mat2[405504:811008], dtype=np.float32) #405504*2 = 811008
dog = np.reshape(dog, (99, -1))
dog_pic = dog[0]

dog_mirror = np.fliplr(dog)
dog = np.array(dog, dtype=np.float32)
dog_mirror = np.array(dog_mirror, dtype=np.float32)

doggy_dog = np.append(dog,dog_mirror)
doggy_dog = np.reshape(doggy_dog,(198,-1))
doggy_dog = doggy_dog.T

# Print the cat and dog arrays and their lengths
print(dog_pic)
print(cat)

woof = np.reshape(dog_pic,(len(dog_pic),-1))

U,S,V_T = svd(kitty_cat, full_matrices=False)

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
    f = lambda i : 51918.40544*i+919.15093
    val = int(f(i))
    eigendogs = np.append(eigendogs,val)

print("Eigendogs")
print(eigendogs)

eigendogs = np.reshape(eigendogs,(64,64)).astype(np.uint8)
cv2.imshow('Eigendog Image', eigendogs)

cv2.waitKey(0)
cv2.destroyAllWindows()