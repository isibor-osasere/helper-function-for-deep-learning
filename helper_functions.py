# importing Required model
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import datetime
import tensorflow_hub as hub
import os
from tensorflow.keras.layers.experimental import preprocessing

# Visualizing our images
def view_random_image (target_dir, target_class, num_of_samples = 1):
    # settingup the target directory
    target_folder = f"{target_dir}/{target_class}"
    
    # Getting a Random image Path
    random_image = random.sample (os.listdir (target_folder), num_of_samples)
    print (f"Image: {random_image}\n\n")
    
    # Visualing the image using matplotlib.image
    for name in random_image:
        img = mpimg.imread (target_folder + "/" + name)
        plt.imshow (img)
        plt.title (f"{target_class.title ()}: {name}")
        plt.axis ("off")
        plt.show ()
        # Getting the image shape
        print (f"Image shape: {img.shape}\n\n")
        
        
# creating a function that visualises augmented train images
def view_augmented_image (train_data, train_data_augmented, batch_size = 32):
    """ A function that visualizes random original and augmented images """
    # Getting sample Data batches
    images, labels = train_data.next ()
    augmented_images, labels = train_data_augmented.next () # only data(images) are augmented, their labels are not

    # show original image and augmented image
    random_number = random.randint (0, batch_size)  # since our batch_size is 32
    print (f"Showing image number: {random_number}")
    plt.figure (figsize = (12, 8))
    plt.subplot (1, 2, 1)
    plt.imshow (images[random_number])
    plt.title ("Original image")
    plt.axis (False)


    plt.subplot (1, 2, 2)
    plt.imshow (augmented_images[random_number])
    plt.title ("Augmented image")
    plt.axis (False)

    
# Creating a function that preprocess our image
def preprocess_image (filename, image_shape = 224, scale = True, channel_num = 3):
    """ A function that reads an image from filename, turns it into
        tensors and resize the image to (image_shape, image_shape, color_channel) """
    # Read in the image file
    img = tf.io.read_file (filename)
    
    # Decode the read file into tensors
    img = tf.image.decode_image (img, channels = channel_num)
    
    # Resize the image
    img = tf.image.resize (img, size = (image_shape, image_shape))
    
    # Scale? yes/no
    if scale:
        img = img/255.   # Rescale the image (get all the values between 0 and 1)
        return (img)
    else:
        return img   # Some models like EfficentNet have scaling already builtin
    #print ("\n\n", img)


def pred_and_plot (model, filename, processed_img, class_name):
    """ A function that predict and visualise our prediction """
    # Making our prediction
    prediction = model.predict (processed_img)
    img = mpimg.imread (filename)
    plt.imshow (img)
    plt.axis (False)
    plt.title ("Image To Be Classified", fontsize = 15)
    plt.show ()
    
    # checking for binary or multiclass
    if len (prediction[0]) > 1:
        proba = np.round (float (prediction.max ()), 2)
        predicted_label = np.argmax (prediction)
    else:
        proba = np.round (float (prediction), 2)
        predicted_label = int (np.round(prediction))
        
    print (f"\nWe Are {proba * 100}% Sure That It's A: {np.array (class_name)[predicted_label]}")

    
# Creating a function that creates TensorBoard Callbacks
def create_tensorboard_callbacks (dir_name, experiment_name):
    # Creates a folder to store the Log files
    log_dir = dir_name + "/" + experiment_name + "/  " + datetime.datetime.now ().strftime ("%Y%m%d-%H%M%S")
    
    # Creating our Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard (log_dir = log_dir)
    print (f"Saving TensorBoard Log files to: {log_dir}")
    
    return tensorboard_callback



# creating a function to create model from a URL
def create_model (model_url, num_classes = 10):
    """
    Takes a Tensorflow Hub URL and creates a keras sequential model with it
    
    Args:
        model_url (str): A tensorflow Hub feature extractor URL
        num_classes (int): Number of output neurons in the output layer,
                           should be equal to the number of target classes
                           
    Returns:
        An uncompiled keras sequential model with model_url as feature extractor
        layer and Dense output layer with num_classes output neurons
    """
    # Download the Pretrained model and save it as a Keras feature_extractor_layer
    feature_extractor_layer = hub.KerasLayer (model_url, 
                                              trainable = False,  # Freezes the already learned patterns
                                              name = "feature_extractor_layer",
                                              input_shape = (224, 224, 3))
    
    # Creating our own model
    model = tf.keras.Sequential ()
    model.add (feature_extractor_layer)
    
    model.add (tf.keras.layers.Dense (num_classes, activation = 'softmax', name = "Output_layer"))
    
    return model


# A function that plots loss curves
def plot_loss_curves (history):
    """ A function that plots loss and accuracy seperately"""
    # Training parameters
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = list(range(len (accuracy)))

    # plotting loss
    plt.plot (epochs, loss, label = "Training_loss")
    plt.plot (epochs, val_loss, label = "val_loss")
    plt.title ("Loss")
    plt.xlabel ("Epochs")
    plt.legend ()

    # plotting accuracy
    plt.figure ()
    plt.plot (epochs, accuracy, label = "Training_accuracy")
    plt.plot (epochs, val_accuracy, label = "val_accuracy")
    plt.title ("Accuracy")
    plt.xlabel ("Epochs")
    plt.legend ()



# Creating a function to compare training histories
def compare_history (original_history, new_history, initial_epochs = 5):
    """
    compares two tensorflow history objects
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    val_acc = original_history.history["val_accuracy"]
    
    loss = original_history.history["loss"]
    val_loss = original_history.history["val_loss"]
    
    # combine original history metrics with new history metrics
    total_acc = acc + new_history.history["accuracy"]
    total_val_acc = val_acc + new_history.history["val_accuracy"]
    
    total_loss = loss + new_history.history["loss"]
    total_val_loss = val_loss + new_history.history["val_loss"]
    
    # make Accuracy Plots
    plt.figure (figsize = (8, 8))
    plt.subplot (2, 1, 1)
    plt.plot (total_acc, label = "Training Accuracy")
    plt.plot (total_val_acc, label = "Val Accuracy")
    plt.plot ([initial_epochs - 1, initial_epochs - 1], plt.ylim (), label = "Start Fine Tuning")
    plt.legend (loc = "lower right")
    plt.title ("Training and Validation Accuracy")
    
    # make Loss plots
    plt.figure (figsize = (8, 8))
    plt.subplot (2, 1, 2)
    plt.plot (total_loss, label = "Training Loss")
    plt.plot (total_val_loss, label = "Val Loss")
    plt.plot ([initial_epochs - 1, initial_epochs - 1], plt.ylim (), label = "Start Fine Tuning")
    plt.legend (loc = "upper right")
    plt.title ("Training and Validation Loss")



def plot_loss_curves (history):
    """ A function that plots loss and accuracy seperately"""
    # Training parameters
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    
    epochs = list(range(len (accuracy)))
    
    # plotting loss
    plt.plot (epochs, loss, label = "Training_loss")
    plt.plot (epochs, val_loss, label = "val_loss")
    plt.title ("Loss")
    plt.xlabel ("Epochs")
    plt.legend ()
    
    # plotting accuracy
    plt.figure ()
    plt.plot (epochs, accuracy, label = "Training_accuracy")
    plt.plot (epochs, val_accuracy, label = "val_accuracy")
    plt.title ("Accuracy")
    plt.xlabel ("Epochs")
    plt.legend ()


# A function that creates a data augmentation layer right into our model
def data_augmentation (input_layer):
    """ A function that creates a data augmentation layer right into our model """

    # Creating the data augmentation layer
    data_augmentation = tf.keras.Sequential ([
        preprocessing.RandomFlip ("horizontal"),
        preprocessing.RandomRotation (0.4),
        preprocessing.RandomZoom (0.2),
        preprocessing.RandomHeight (0.3),
        preprocessing.RandomWidth (0.2),
        #preprocessing.Rescale (1./255) # used for models like ResNet50v2 but EfficientNet have rescaling already buildt-in
    ], name = "data_augmentation layer")




# A function that creates a checkpoint callback
def checkpoint_callback (checkpoint_path):
    """A function that creates a checkpoint callback"""
    
    # Create a ModelCheckpoint callback that saves the model's weights only
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint (filepath = checkpoint_path,
                                                              save_weights_only = True,
                                                              save_best_only = True,
                                                              monitor = "val_accuracy",
                                                              verbose = 1)



# matplotlib Autolabeling
def autolabel (rects):
    for rect in rects:
        height = rect.get_height ()
        ax.text (rect.get_x () + rect.get_width ()/2., 1.05 * height, '%d' % int (height), ha = "center", va = "bottom")























