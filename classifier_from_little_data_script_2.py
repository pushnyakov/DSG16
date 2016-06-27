from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import h5py
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np


# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 7200 / 2
nb_validation_samples = 800
nb_epoch = 10


model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(1, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

'''
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
'''

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))



model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(4))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
                                   #samplewise_std_normalization=True,
                                   #samplewise_center=True,
                                   #rotation_range=5,
                                   #width_shift_range=0.2,
                                   #height_shift_range=0.2,
                                   rescale=1./255,
                                   #shear_range=0.2,
                                   #zoom_range=0.2,
                                   #horizontal_flip=True)
                                   )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(#samplewise_std_normalization=True,
                                  #samplewise_center=True,
                                  rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=64,
                                                    color_mode="grayscale",
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
                                                        
                                                        validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=64,
                                                        color_mode="grayscale",
                                                        class_mode='categorical')

model.fit_generator(
                    train_generator,
                    samples_per_epoch=nb_train_samples,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=nb_validation_samples)

model.save_weights('4nd_try.h5', overwrite=True)

model.load_weights('4nd_try.h5')

for i in range(4):
    mypath = '/Users/philipppushnyakov/data/validation/' + str(i+1) + '/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    s = open('test_valid.txt', 'wt')
    cnt = 0
    for f in onlyfiles:
        im_name = mypath + f
        try:
            im = cv2.cvtColor(cv2.resize(cv2.imread(im_name), (64, 64)).astype(np.float32), cv2.COLOR_BGR2GRAY)
        except KeyboardInterrupt: raise
        except:
            print im_name
            continue
        #print im.shape
        #im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
            #for j in range(1):
            #im[:,j,:,:] -= np.mean(im[:,j,:,:])
            #im[:,j,:,:] /= np.std(im[:,j,:,:])
        #im -= np.mean(im)
        #im /= np.std(im)
        out = model.predict(im)
        #print out
        if np.argmax(out) + 1 == i + 1:
            cnt += 1
        s.write(f.split('.')[0] + ',' + str(np.argmax(out) + 1) + '\n')

    print(cnt)


def prepare_submission():
    mypath = '/Users/philipppushnyakov/data/test/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    s = open('submission.csv', 'wt')
    s.write('Id,label' + '\n')
    for f in onlyfiles:
        im_name = mypath + f
        try:
            im = cv2.resize(cv2.imread(im_name), (64,64))
            #im = cv2.cvtColor(cv2.resize(cv2.imread(im_name), (64, 64)).astype(np.float32), cv2.COLOR_BGR2GRAY)
        except KeyboardInterrupt: raise
        except:
            print im_name
            continue
        #print im.shape
        #im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        im = np.expand_dims(im, axis=0)
        #for j in range(1):
        #im[:,j,:,:] -= np.mean(im[:,j,:,:])
        #im[:,j,:,:] /= np.std(im[:,j,:,:])
        #im -= np.mean(im)
        #im /= np.std(im)
        out = model.predict(im)
        #print out
        s.write(f.split('.')[0] + ',' + str(np.argmax(out) + 1) + '\n')

prepare_submission()