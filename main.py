import os

import tensorflow as tf
import logging

from keras.callbacks import EarlyStopping

tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import cv2


main_path = './Dataset/'
img_size = (100, 100) #(64, 64)
batch_size = 64

from keras.utils import image_dataset_from_directory



Xtrain, Xval = image_dataset_from_directory(main_path, subset='both', validation_split=0.3, image_size=img_size, batch_size=batch_size, seed=123)

val_batches = tf.data.experimental.cardinality(Xval)
Xtest = Xval.take((2*val_batches) // 3)
Xval = Xval.skip((2*val_batches) // 3)



classes = Xtrain.class_names
print(classes)

class_dis = [len(os.listdir(main_path + name)) for name in classes]

plt.bar(classes, class_dis)
#plt.show()


N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')

    #plt.show()


from keras import layers
from keras import Sequential
data_augmentation = Sequential([ layers.RandomFlip("horizontal", input_shape=(img_size[0], img_size[1], 3)), layers.RandomRotation(0.25), layers.RandomZoom(0.1), ])

N = 10
plt.figure()
for img, lab in Xtrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

    #plt.show()


from keras import Sequential
from keras import layers
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy


num_classes = len(classes)
model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(100, 100, 3)),
    layers.Conv2D(25, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(50, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(100, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(260, activation='relu'),
    layers.Dense(130, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


model.summary()
model.compile(Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

history = model.fit(Xtrain, epochs=50, validation_data=Xval, callbacks=[es],verbose=0)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
#plt.show()



labels = np.array([])
pred = np.array([])
for img, lab in Xval:
    labels = np.append(labels, lab)
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))


from sklearn.metrics import accuracy_score
print('Tačnost modela na validacionom skupu je: ' + str(100*accuracy_score(labels, pred)) + '%')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
cmDisplay.plot()
#plt.show()





labels2 = np.array([])
pred2 = np.array([])
for img, lab in Xtest:
    labels2 = np.append(labels2, lab)
    pred2 = np.append(pred2, np.argmax(model.predict(img, verbose=0), axis=1))

print('Tačnost modela testnom skupu je: ' + str(100*accuracy_score(labels2, pred2)) + '%')
cm2 = confusion_matrix(labels2, pred2, normalize='true')
cmDisplay2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes)
cmDisplay2.plot()


N = 10
plt.figure()
predictions = model.predict(Xtest, verbose=0)
for img, lab in Xtest.take(1):
    for i in range(N):
        maxPred = max(predictions[i])
        guess = np.where(predictions[i] == maxPred)
        plt.subplot(2, int(N / 2), i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        if (str(guess) == classes[lab[i]]):
            plt.title(classes[lab[i]])

        else:
            plt.title(str(guess[0]))
        plt.axis('off')

    plt.show()
