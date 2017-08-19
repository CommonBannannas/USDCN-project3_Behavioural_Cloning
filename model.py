import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D
import os
import matplotlib.image as mpimg
import cv2
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

old_path = '/home/sergio/workspace/sim_data/'

def fix_path(data_df):
    cols = data_df.columns.tolist()
    for col in data_df.columns:
        try:
            data_df[col] = data_df[col].apply(lambda x: x.replace(old_path, ''))
        except:
            pass
    return data_df

def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)

    return image

def augment_images(data_dir, center, left, right, steering_angle):
    # Choose an image from left, center or right and adjust steering angle
    choice = np.random.choice(3)
    if choice == 0:
        image = mpimg.imread(os.path.join(data_dir, left.strip()))
        steering_angle += 0.2
    elif choice == 1:
        image = mpimg.imread(os.path.join(data_dir, right.strip()))
        steering_angle -= 0.2
    elif choice ==2:
        image = mpimg.imread(os.path.join(data_dir, center.strip()))

    # make a random flip on the image
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle

    return image, steering_angle

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    # Generate training image 
    images = np.empty([batch_size, IM_HEIGHT, IM_WIDTH, IM_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            if is_training and np.random.rand() < 0.6:
                # augment data when in training
                image, steering_angle = augment_images(data_dir, center, left, right, steering_angle)
                
                
            else:
                # chooses image from center
                image = mpimg.imread(os.path.join(data_dir, center.strip()))
            
            
            # add image and steering angle
            images[i] = add_shadow(augment_brightness(image))
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers


np.random.seed(42)
IM_HEIGHT = 160
IM_WIDTH = 320
IM_CHANNELS = 3

# load data
data_dir = './data2/'
test_size = .25

data_df = pd.read_csv(os.path.join(data_dir,'driving_log_final.csv'))
data_df.columns = ['center','left','right','steering','throttle','break','speed']

data_df = fix_path(data_df)

X = data_df[['center', 'left', 'right']].values
y = data_df['steering'].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)


print(data_df.head())
print('----------------------------')
print(len(X_train), len(X_valid))
print('----------------------------')
print(len(X_train) + len(X_valid))
print('----------------------------')
# build keras model
INPUT_SHAPE = (IM_HEIGHT, IM_WIDTH, IM_CHANNELS)
keep_prob = .55
# nvidia's network architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/127.5-1.0))

model.add(Conv2D(24, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(36, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(48, 5, activation='elu', strides=(2, 2)))
model.add(Conv2D(64, 3, activation='elu'))
model.add(Conv2D(64, 3, activation='elu'))

model.add(Dropout(keep_prob))

model.add(Flatten())

model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))

model.summary()

# train model
learning_rate = 1.0e-4
batch_size = 64
steps_per_epoch = 500
nb_epoch = 10

model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

model.fit_generator(batch_generator(data_dir, X_train, y_train, batch_size, True),
                    steps_per_epoch,
                    nb_epoch,
                    max_queue_size=1,
                    validation_data=batch_generator(data_dir, X_valid, y_valid, batch_size, False),
                    validation_steps=len(X_valid),
                    verbose=1) 
print('saving...')
model.save('model.h5')





