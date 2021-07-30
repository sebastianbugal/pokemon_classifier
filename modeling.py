from PIL import Image
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GaussianNoise, Conv2D, MaxPool2D, Input
from tensorflow.keras.layers.experimental.preprocessing import RandomTranslation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.initializers import RandomNormal
import pandas as pd
import os
import numpy as np
from operator import itemgetter 

img_size = 128
'''
The code chunk below is the initial investigation of the different 
types of pokemon using one hot encoding to identify between all 
different types. Unfortunately this proved to have very low accuracy rates and I 
therefore decided to simplify the problem. Uncomment the below 
block to try out the one hot encoded method.
'''
# one_hot_key_old = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water']
# one_hot_key = [ 'Fire', 'Water']
# def preprocess_data():
#     arr = []
#     label = []
#     path = "images"
#     dirs = os.listdir(path)
#     one_hot = pd.DataFrame(columns=one_hot_key)
#     for item in dirs:
#         if (item == '.DS_Store'):
#             continue
#         img = image.load_img('images/' + item, target_size=(img_size, img_size),
#                                 color_mode="rgb")  # loading image and then convert it into grayscale and with it's target size
#         X_test = image.img_to_array(img)  # convert image into array
#         X_test = image.random_rotation(X_test, 25)
#         try:
#             temp = None
#             temp = np.zeros((1,len(one_hot_key)))
#             has_one = False
#             for i in item.split('.jp')[0].split('_')[1:]:
#                 try:
#                     temp[0][one_hot_key.index(i)]=1
#                     has_one=True
#                 except:
#                     if(has_one):
#                         continue
#                     else:
#                         raise ValueError('Error')
#             label.append(temp[0].astype('float32'))
#             arr.append(X_test)
#         except:
#             continue
#     return arr, label   

def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements. This function is for the one hot
    encoding to identify the different types of pokemon. '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            index_pos = list_of_elems.index(element, index_pos)
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list

def preprocess_data():
    arr = []
    label = []
    path = "images"
    dirs = os.listdir(path)
    for item in dirs:
        if (item == '.DS_Store'):
            continue
        img = image.load_img('images/' + item, target_size=(img_size, img_size),
                                color_mode="rgb")  # loading image and then convert it into grayscale and with it's target size
        X_test = image.img_to_array(img)  # convert image into array
        X_test = image.random_rotation(X_test, 45)
        temp = None
        for i in item.split('.jp')[0].split('_')[1:]:
            if(i=='Fire'):
                temp=1
            elif(i=='Water'):
                temp=0
        if(temp!=None):           
            label.append(temp)
            arr.append(X_test)
    return arr, label   

def identify_type(arr_one_hot):
    return itemgetter(*get_index_positions(arr_one_hot.tolist(),1))(one_hot_key)

def predict_save(model,number_of_predictions):
    '''This method is to provide some visual understanding of what the network seems to predict.'''
    arr = []
    label = []
    path = "images"
    dirs = os.listdir(path)
    for item in dirs:
        if (item == '.DS_Store'):
            continue
        img = image.load_img('images/' + item, target_size=(img_size, img_size),
                                color_mode="rgb") 
        img_arr = image.img_to_array(img)  # convert image into array
        temp = None
        for i in item.split('.jp')[0].split('_')[1:]:
            if(i=='Fire'):
                temp=1
            elif(i=='Water'):
                temp=0
        if(temp!=None):           
            arr.append({'name':item.split('_')[0], 'img': img_arr})
    
    for i in range(number_of_predictions):
        random_image = random.choice(arr)
        outcome = model.predict(np.array(random_image['img']).reshape(1,128,128,3))
        plt.imsave(f'images_prediction/{random_image["name"]}_{str(outcome[0][0])}.jpg', random_image['img']/255)


def model():
    # ran=RandomNormal(stddev=0.02)
    model=Sequential()
    
    model.add(RandomTranslation(input_shape=(img_size,img_size,3), 
        height_factor=(-0.15, 0.15), 
        width_factor=(-0.15,0.15), 
        fill_mode = "nearest" ))

    model.add(Conv2D(32, kernel_size=5, activation='relu', padding='same', input_shape=(img_size,img_size,3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(GaussianNoise(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))     
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()
    img = Input(shape=(img_size,img_size,3))
    d = model(img)
    mod = Model(img,d)
    return mod

def main():
    pre_data = preprocess_data()
    model_ = model()
    model_.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.000001), metrics=['accuracy'] )
    
    X = np.array(pre_data[0])
    y = np.array(pre_data[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    print(len(X_train))
    history  = model_.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=1000,
        validation_data=(X_test, y_test),
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    predict_save(model_, 10)
if __name__ == "__main__":
    main()
