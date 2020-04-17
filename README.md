# Neural Networks for Diagnosing Pneumonia

## Business Problem

Recognition images for disease diagnosis remains a challenge in the healthcare industry. When there are intensive X-Ray images that need to be used for identifying if people have pneumonia in a short amount of time, it requires high workload from doctors or experts. For example, during covid-19, some countries use X-ray images as one source to diagnose if someone is infected or recovered. The hospital could apply AI technology for the image recognition to reduce the burden of the doctors for the recovering check. 

Image recognition could help hospitals more efficiently organize and manage their documents and resources. 

Mostly the interpretation of medical data is being done by medical experts. In terms of image interpretation by human experts, it is quite limited due to its subjectivity, complexity of the image, extensive variations exist across different interpreters, and fatigue. A high accuracy classification deep learning model could complement the human expertise. 

A high accuracy recognition model could be used in rural areas that lack expertise for diagnosis. 

This model could be used as a pretrained model that could  apply to other medical images.

## Project Goal

The goal of this project is to build a deep learning algorithm that could accurately diagnose pneumonia. More specifically, the algorithm would have high performance on accurately classifying true positive and lower rate on false negative. 

## Data Collection & Processing

Data for training and testing the model could be downloaded from Kaggle and Mendeley.

Kaggle: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia  
Mendeley: https://data.mendeley.com/datasets/rscbjbr9sj/3

The data stored in Kaggle is a subset of the data stored in Mendeley. The size of the dataset in Kaggle is 1.9 GB, and in Mendeley is 7.9 GB. 

The datasets on Kaggle and Mendeley are both categorized into three folders already: ‘train’, ‘test’, ‘val’. In each folder, it includes two sub-folders as ‘normal’ and ‘pneumonia’. Therefore, there is not much proprecessing work to clean the data.

We used the subset from kaggle to train the model first. Once we got a satisfying model and adjusted the hyperparameters to our liking, we moved to use the large dataset from mendeley to retrain the model. The accuracy score was substantially improved after transferring to the large dataset.

#### Scaling & Reading images
    - Create train_datagen and validation_datagen using keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    - Read the images using keras.flow_from_directory
    - Set image size and batch size

## Build the Model

We initially used transferred learning with the MobileNetV2 model from keras.applications. Nevertheless, the validation accuracy is only around 0.7 by running 5-20 epochs. We also tried another pretrained algorithm VGG16 from keras.application. The result was similar to MobileNetV2. The transferred learning model took a long time to run due to the complexity but did not perform well in our case.

We then moved to apply basic convolutional neural networks (CNN) with multiple hidden layers and dropouts to prevent overfitting on the training set. After tuning the models, we reached a validation accuracy to 0.89. 

Once the model performance was satisfying, we moved to use the large dataset downloaded from Mendeley to train the model again. The results of the final model fitting with the large dataset is fairly good with the validation accuracy score as 0.96. 

## Evaluation

Accuracy: 0.789  
Precision: 0.856  
Recall: 0.997  
F-1: 0.749  

Our final model has a satisfactory performance on accuracy and precision, which means it is pretty reliable when it predicts pneumonia. It also has an excellent performance on recall, which means it minimizes false negative very well. Nevertheless, the false positive rate is relatively high. The next step is to do more research to reduce the cases of false positive.  
