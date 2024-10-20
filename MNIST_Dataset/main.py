from keras.api.datasets import mnist
from keras import models 
from keras import layers
from keras.src.utils import to_categorical
from PIL import Image
import PIL.ImageOps
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Train images shape:", train_images.shape)
print("Train labels length:", len(train_labels)) 
print("Train labels:", train_labels)             

print("Test images shape:", test_images.shape)    
print("Test labels length:", len(test_labels))   
print("Test labels:", test_labels)    

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) 
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)

#### MY IMAGE ####
image = Image.open("image.jpg")
image = image.convert('L')
image = image.point(lambda p: 255 if p > 128 else 0)
image = image.resize((28, 28))
inverted_image = PIL.ImageOps.invert(image)
inverted_image.save("I.jpg")

# Convert the image to a NumPy array
image_array = np.array(inverted_image)

# Normalize the image (scale pixel values to 0-1 range)
image_array = image_array.astype('float32') / 255

image_array = image_array.reshape((1, 28 * 28))

predictions = network.predict(image_array)

# Get the index of the highest probability class
predicted_label = np.argmax(predictions)

print(f"The model predicts the image is a '{predicted_label}'")
