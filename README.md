# Introduction

In this project, we use SVD and PCA techniques to perform transformations on a dataset consisting of images of cats and dogs. The objective is to explore the process of facial recognition and gain insights into the underlying patterns and characteristics of the images.

# Transformations
We performed several transformations on the dataset:

1. Average Images: We computed the average cat and dog images by calculating the mean pixel values across all images in each class. These average images provide a representation of the typical appearance of cats and dogs in the dataset.
''

2. Symmetrized Images: We created symmetrized cat and dog images by averaging each image with its horizontally flipped counterpart. This process aims to enhance symmetrical features in the images and investigate the impact on facial recognition.

3. Doubling the Dataset: To obtain clear eigencat and eigendog images, we doubled the dataset by replicating each image. This allows for better characterization of the principal components and understanding of the variations within each class.
