import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_images = train_images/255.0
test_images = test_images/255.0

# print(train_images[7])
#
# plt.imshow(test_images[7], cmap=plt.cm.binary)
# plt.show()

#creating a model

model = keras.Sequential([ # sequential just means adding each of these layers in order
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(128, activation="relu"), # dense means full connected hidden layer
    keras.layers.Dense(10, activation="softmax") # picks values of each neuron so that all of those values add up to 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# time to train the model!
model.fit(train_images, train_labels, epochs=5) # epochs = how many times the model is going to see the information (ie the same image)


# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Tested Accuracy: ", test_acc)

# using the model to make predictions

prediction = model.predict([test_images[7]])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediciton " + class_names[np.argmax(prediction[i])])
    plt.show()
