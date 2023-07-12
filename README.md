# Introduction

In this project, we use SVD and PCA techniques to perform transformations on a dataset consisting of images of cats and dogs. The objective is to explore the process of facial recognition and gain insights into the underlying patterns and characteristics of the images.

# Transformations
We performed several transformations on the dataset:

1. Average Images: We computed the average cat and dog images by calculating the mean pixel values across all images in each class. These average images represent the typical appearance of cats and dogs in the dataset.

2. Symmetrized Images: We created symmetrized cat and dog images by averaging each image with its horizontally flipped counterpart. This process aims to enhance symmetrical features in the images and investigate the impact on facial recognition.

3. Doubling the Dataset: To obtain clear eigencat and eigendog images, we doubled the dataset by replicating each image. This allows for better characterization of the principal components and an understanding of the variations within each class.

# Results
During our analysis, we made the following observations:

* Error in Projection: When projecting a dog image onto the cat image class, we encountered greater errors compared to projecting a cat image onto the dog image class. This suggests that the dog images may share more visual features with cats than vice versa.

* Dataset Size: While working with a smaller dataset resulted in quicker runtime, we recognized the need for a larger dataset. A larger dataset would provide more diverse cat and dog images, including variations in color, poses, and lighting conditions. This would improve the model's accuracy and robustness.

# Installation
To use this project, follow these steps:

1. Clone the repository:
   <pre>git clone https://github.com/JimmySitompul/catdog-image-recognition.git</pre>
2. Navigate to the project directory: <pre>cd catdog-image-recognition</pre>
3. Install the required dependencies: <pre>pip install -r requirements.txt</pre>

# Usage
1. Run the provided scripts to perform the SVD and PCA transformations on the dataset.

2. Analyze the generated average cat and dog images, symmetrized images, and eigencat and eigendog images.

3. Evaluate the error in projection between cat and dog images.

4. Consider the impact of dataset size on runtime and explore the possibility of incorporating a larger, more diverse dataset.
