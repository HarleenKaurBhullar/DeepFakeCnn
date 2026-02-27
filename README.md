# Deepfake Image Detection using CNN

## Project Overview

This project implements a deepfake image classification system using a Convolutional Neural Network (CNN). The objective is to study whether spatial features learned by a lightweight CNN can help distinguish between real and deepfake images. The focus of this project is on correct machine learning workflow, dataset handling, and transparent evaluation rather than aggressive metric optimization.

---

## Dataset

* Source: Hugging Face dataset `prithivMLmods/Deepfake-vs-Real-60K`
* Original dataset size: Approximately 60,000 images
* Classes:

  * Label 0: Real images
  * Label 1: Deepfake images

### Dataset Selection

To ensure class balance and manageable computational requirements:

* 10,000 real images were randomly sampled
* 10,000 deepfake images were randomly sampled

This resulted in a balanced dataset of 20,000 images used for training and evaluation.

---

## Data Preprocessing

Image preprocessing was performed using TorchVision transforms with the following steps:

* Resize images to 256 x 256
* Center crop to 224 x 224
* Convert images to tensors
* Normalize using ImageNet statistics

  * Mean: [0.485, 0.456, 0.406]
  * Standard deviation: [0.229, 0.224, 0.225]

Preprocessing was applied in batches of 32 using Hugging Face `map()` with multiprocessing enabled. The original image column was removed after transformation. The processed dataset was saved using `save_to_disk()` and also exported in Parquet format to allow reuse without repeating preprocessing.

---

## Dataset Split

After preprocessing, the dataset was split into:

* Training set: 80 percent
* Test set: 20 percent

The split was performed after preprocessing to maintain consistency in input formatting and to avoid redundant preprocessing steps.

---

## Data Pipeline

The Hugging Face dataset was converted into a TensorFlow `tf.data.Dataset` using native integration. The pipeline includes:

* Lazy loading to avoid loading the entire dataset into memory
* Batching and prefetching for efficient training
* Channel order conversion from (3, 224, 224) to (224, 224, 3) for TensorFlow compatibility

---

## Model Architecture

A custom CNN was implemented using TensorFlow and Keras with the following structure:

* Three convolutional layers with filter sizes 32, 64, and 128
* ReLU activation after each convolution
* MaxPooling after each convolutional layer
* Fully connected layer with 128 units and ReLU activation
* Dropout layer with a rate of 0.5 to reduce overfitting
* Output layer with a single neuron and sigmoid activation for binary classification

Loss function: Binary cross entropy
Optimizer: Adam

The architecture was intentionally kept simple to establish a strong baseline before experimenting with deeper or pretrained models.

---

## Model Training

* Epochs: 5
* Batch size: 32
* Validation performed using the held out test split
* Metrics tracked: Training loss, validation loss, and accuracy

Loss curves were plotted to observe training dynamics and potential overfitting.

---

## Evaluation

The model was evaluated on the test dataset. Evaluation focused on understanding loss behavior and general trends rather than relying solely on accuracy. This cautious approach is important given the controlled dataset size and potential similarity among samples.


## Conclusion

This project demonstrates an end to end deep learning workflow for image based deepfake detection with an emphasis on balanced data selection, efficient preprocessing, and responsible evaluation. It serves as a foundation for further exploration into more robust and generalizable deepfake detection systems.
