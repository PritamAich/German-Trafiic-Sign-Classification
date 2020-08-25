# German-Trafiic-Sign-Classification 

The German Traffic Sign Benchmark is a multi-class, single-image classification challenge held at the International Joint Conference on Neural Networks (IJCNN) 2011. We cordially invite researchers from relevant fields to participate: The competition is designed to allow for participation without special domain knowledge. Our benchmark has the following properties: 

  

1. Single-image,  

2. Multi-class classification problem 

3. More than 40 classes 

4. More than 50,000 images in total 

5. Large, lifelike database 

 

The dataset was taken from kaggle . [Click here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

#### Note: The dataset was too large to upload in github. So, the link is provided above.

### Project architecture:

**Data Collection**

**Visualization**

  - Sample images
  - Joint plots
      
**Data Preprocessing**

  - Resizing images
  - Converting images into numpy array
  - Rescaling Images
      
**Model Building**

  - Data Spliting
  - CNN model creation
  - Training model on train set images
      
**Testing**
   
  - Tesing on test images
      
      
## 1. Data collection:

Download the dataset from [here](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

  1. Imported labels.csv file to see the meaning of each class associated with this dataset
  2. Assigned the paths for train and test images

## 2. Visualization:

  1. Data Visualization on some of the sample images from test dataset using 'matplotlib' to get a grasp of the images we are dealing with. 
     Some of those images are shown below.
  2. Extracted dimensions of each image from training set and plotted them on a jointplot using 'seaborn'.
      
      
## 3. Data Preprocessing:

  1. Imported images from training path, used Image method from PIL package for image retrieval.
  2. Converted images to numpy array and stored them.
  3. Scaled the images so that the pixel values of each image remain between 0 and 1. This is also known as normalization.
     **Normalization is a process that changes the range of pixel intensity values.**
  4. And lastly used countplot from seaborn to count the number of images in each distinct class.
  5. Saved the scaled images data and labels for future use in the model.
      
#### Note: The 'Training.npy' file saved during during this process was around 2.2gb size which was not possible to upload in github. Run the notebook file and it will automatically save that file in local.

## 4. Model creation:
      
  1. Loaded the training images and labels saved as numpy files.
  2. Used train_test_split module from sklearn package to solit the training data into train and validation sets. 
     The split ratio was 80% training and 20% validation data.
  3. Categorized the labels of both train and validation set using one one-hot encoding technique. 
     Used 'to_categorical' method from keras package.
  4. Created a CNN model with 3 layers, used MaxPool2D for pooling layers(pooling size = (2,2)) and dropout layers with a dropout rate of 0.5 for each layer.
     Then flattened the layers and finally added a dense layer. Used 'relu' activation function for each
  5. Used 'softmax' activation function for the final output layer.
  6. During compilation, the following were used:
      - Loss = 'sparse_categorical_crossentropy"
      - Optimizer = 'adam'
      - Metrics = 'accuracy'
  7. Trained the model with epochs = 15, used early stopping to prevent overfitting.
          
          model = Sequential()

          #1st layer
          model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape = x_train.shape[1:], activation = 'relu', padding = 'same'))
          model.add(MaxPool2D(pool_size=(2,2)))
          model.add(Dropout(0.5))

          #2nd layer
          model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
          model.add(MaxPool2D(pool_size=(2,2)))
          model.add(Dropout(0.5))

          #3rd layer
          model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))
          model.add(MaxPool2D(pool_size=(2,2)))
          model.add(Dropout(0.5))

          model.add(Flatten())

          #Dense layer
          model.add(Dense(128, activation = 'relu'))
          model.add(Dropout(0.5))

          #Output layer
          model.add(Dense(43, activation = 'softmax'))

  8. Achieved an accuracy of 99.4% on the validation set.
  9. Lastly saved the model in 'h5' format.

## 5. Testing on test images:
      
  1. Loaded our CNN model.
  2. Imported the test images, normalized them by scaling. 
     For this, implemented a function that will take image folder path as an argument and then resize and scale the images for the model. 
     This function can be later used for any future testing purpose.
  3. Tested these images on our model and predicted their classes.
  
                  precision    recall  f1-score   support

                     0       0.97      1.00      0.98        60
                     1       0.98      0.99      0.99       720
                     2       0.98      0.99      0.99       750
                     3       1.00      0.94      0.97       450
                     4       1.00      0.98      0.99       660
                     5       0.95      0.98      0.96       630
                     6       1.00      0.88      0.94       150
                     7       0.99      0.97      0.98       450
                     8       0.98      0.98      0.98       450
                     9       0.96      1.00      0.98       480
                    10       1.00      1.00      1.00       660
                    11       0.91      0.99      0.95       420
                    12       0.99      0.95      0.97       690
                    13       1.00      1.00      1.00       720
                    14       1.00      1.00      1.00       270
                    15       0.97      1.00      0.98       210
                    16       1.00      0.99      1.00       150
                    17       1.00      0.90      0.95       360
                    18       0.94      0.95      0.94       390
                    19       0.95      1.00      0.98        60
                    20       0.79      1.00      0.88        90
                    21       0.96      0.73      0.83        90
                    22       0.99      0.99      0.99       120
                    23       0.93      0.98      0.95       150
                    24       0.99      0.93      0.96        90
                    25       0.98      0.98      0.98       480
                    26       0.94      0.93      0.94       180
                    27       0.86      0.50      0.63        60
                    28       0.97      0.98      0.97       150
                    29       0.89      1.00      0.94        90
                    30       0.98      0.71      0.83       150
                    31       0.94      0.96      0.95       270
                    32       0.62      1.00      0.77        60
                    33       0.99      1.00      0.99       210
                    34       1.00      0.99      1.00       120
                    35       0.99      1.00      1.00       390
                    36       0.98      1.00      0.99       120
                    37       0.95      1.00      0.98        60
                    38       0.97      1.00      0.98       690
                    39       0.97      0.98      0.97        90
                    40       0.88      0.97      0.92        90
                    41       1.00      0.72      0.83        60
                    42       0.93      1.00      0.96        90

              accuracy                           0.97     12630
             macro avg       0.95      0.95      0.95     12630
          weighted avg       0.97      0.97      0.97     12630

  4. Achieved an overall accuarcy of 97% on the test images which is quite good.
