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

      1. Data Visualization on some of the sample images from test dataset using 'matplotlib' to get a grasp of the images we are dealing with. Some of those images are shown            below.
      2. Extracted dimensions of each image from training set and plotted them on a jointplot using 'seaborn'
      
## 3. Data Preprocessing:

      1. Imported images from training path, used Image method from PIL package for image retrieval.
      2. Converted images to numpy array and stored them.
      3. Scaled the images so that the pixel values of each image remain between 0 and 1. This is also known as normalization.
         **Normalization is a process that changes the range of pixel intensity values.**
      4. And lastly used countplot from seaborn to count the number of images in each distinct class.
      5s. Saved the scaled images data and labels for future use in the model.
      
#### Note: The 'Training.npy' file saved during during this process was around 2.2gb size which was not possible to upload in github. Run the notebook file and it will automatically save that file in local.

## 4. Model creation:
      
      1. Loaded the training images and labels saved as numpy files.
      2. Used train_test_split module from sklearn package to solit the training data into train and validation sets. The split ratio was 80% training and 20% validation data.
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
      7. Achieved an accuracy of 99.4% on the validation set.
      8. Lastly saved the model in 'h5' format.

## 5. Testing on test images:
      
      1. Loaded our CNN model.
      2. Imported the test images, normalized them by scaling. For this, implemented a function that will take image folder path as an argument and then resize and scale the              images for the model. This function can be later used for any future testing purpose.
      3. Tested these images on our model and predicted their classes.
      4. Achieved an overall accuarcy of 97% on the test images which is quite good.
