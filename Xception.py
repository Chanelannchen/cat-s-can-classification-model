from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception

# load dataset
train_path = '/DATASET/train'
test_path = '/DATASET/test'

batch_size = 16
image_size = (224, 224)
epoch = 15

# load retrained model
model = Xception(include_top=False,
                 weights='imagenet',
                 input_shape=(224, 224, 3))

# set output
x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(20, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# set early stop,checkpoint,reducing learning rate
estop = EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1)
checkpoint = ModelCheckpoint('Xception_checkpoint.h5', verbose=1,
                monitor='val_loss', save_best_only=True,
                mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                patience=5, mode='min', verbose=1,
                min_lr=1e-4)

# set imagedatagenerator
train_datagen = ImageDataGenerator(rescale= 1.0/255,
                   rotation_range=20,
                   width_shift_range=0.2,
                   height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2)
              
test_datagen = ImageDataGenerator(rescale= 1.0/255)
  
  
train_generator = train_datagen.flow_from_directory(train_path,
                            target_size=image_size,
                            class_mode='categorical',
                            shuffle=True,
                            batch_size=batch_size)
test_generator = test_datagen.flow_from_directory(test_path,
                            target_size=image_size,
                            class_mode='categorical',
                            shuffle=False,
                            batch_size=batch_size)

# train model
history = model.fit_generator(train_generator,
                epochs=epoch, verbose=1,
                steps_per_epoch=train_generator.samples//batch_size,
                validation_data=test_generator,
                validation_steps=test_generator.samples//batch_size,
                callbacks=[checkpoint, estop, reduce_lr])
model.save('./Xception.h5')
print('saved Xception.h5')


# -----------------------------------------------------------------

# check accuracy
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.grid()
plt.savefig('./acc.png')
plt.show()

# check loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.grid()
plt.savefig('loss.png')
plt.show()
