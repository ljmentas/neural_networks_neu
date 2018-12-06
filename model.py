from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 256,256

train_data_dir = '/home/ubuntu/data/128/training_set'
validation_data_dir = '/home/ubuntu/data/128/validation_set'
test_data_dr = '/home/ubuntu/data/128/test_set'
nb_train_samples = 2198
nb_validation_samples = 573
epochs = 200
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height,1)

model = Sequential()
model.add(Conv2D(16,(1, 1), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16,(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(32, (1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
adam = optimizers.Adam(lr = 0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator( rescale=1. / 255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

test1_datagen = ImageDataGenerator(rescale = 1. /255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode = 'categorical',
    color_mode = 'grayscale',
    shuffle = True)

x,y=train_generator.next()


validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode= 'categorical',   
    color_mode='grayscale',
    shuffle = False)
#checkpointer = [ModelCheckpoint(filepath='/home/ubuntu/data/weights-improved-temp2.hdf5',monitor='val_loss',
 #                              verbose=1, save_best_only=True, mode = 'min'),
  #              EarlyStopping(monitor='val_loss', min_delta = 0.0000001,verbose=1,patience=200,mode='min')]
checkpointer = [ModelCheckpoint(filepath='/home/ubuntu/data/weights-improved-plot.hdf5',monitor='val_loss',
                               verbose=1, save_best_only=True, mode = 'min'),
                EarlyStopping(monitor='val_loss', min_delta = 0.0000001,verbose=1,patience=100,mode='min')]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks = checkpointer)
#Plot of Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('Valid_Accuracy.png')

#Plot of Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('Valid_Loss.png')

model.load_weights('/home/ubuntu/data/weights-improved-temp.hdf5')

test_generator = test1_datagen.flow_from_directory(
        test_data_dr,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='categorical',  # only data, no labels
        
        color_mode='grayscale',
        shuffle=False)  # keep data in same order as labels

#probabilities = model.predict_generator(test_generator,90)
#print(probabilities)
test_generator.reset()
model.evaluate_generator(validation_generator,verbose = 0,steps =90)

test_generator.reset()

loss,accuracy = model.evaluate_generator(test_generator,verbose = 0, steps = 90)
print('Accuracy is',accuracy,'Loss is',loss)

pred = model.predict_generator(test_generator,verbose =1,steps =90)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
for k in predictions:
	print (k)

loss,accuracy = model.evaluate_generator(test_generator,verbose = 0, steps = 90)
print('Accuracy is',accuracy,'Loss is',loss)
'''
print(model.metrics_names)
from sklearn.metrics import confusion_matrix

y_pred = probabilities > 0.5
print(len(y_pred))
for test in y_pred:
    print(test)
print(y_pred)
confusion_matrix(y_true, y_pred)
