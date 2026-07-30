# YogaFlow

YogaFlow is an AI-based system for recognizing yoga poses based on computer vision and deep learning. The application uses the webcam to recognize a user’s movements. The system then identifies the user’s position and yoga pose type. The objective of the project was to create an AI assistant for practicing yoga through yoga pose recognition in real-time.


## The Algorithm

YogaFlow uses a computer vision and deep learning pipeline for detecting yoga poses. The algorithm consists of the following steps:

1. Dataset preparation

2. Model training

3. Validation and Testing

4. Real-time pose recognition

In the first step, the yoga pose dataset is separated into the following folders using the dataset_split.py script

train/

val/

test/

Then, the train.py script is used to load the training set and train the model. The algorithm learns to detect yoga poses based on pixel patterns of images.

After model training, the validation set is used to evaluate the model quality and prevent overfitting. Finally, the test.py script is used to test the trained model on images from the test set.


https://youtu.be/8Bjge8eo8FQ
