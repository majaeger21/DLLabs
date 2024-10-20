# Lab 4 - Convolutional Neural Networks (CNNs)
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model (modified model architecture by increasing depth and complexity and included 
# L2 regularization and dropout to help reduce overfitting)
model = models.Sequential()
# First convolutional block
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), 
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
# Second convolutional block
model.add(layers.Conv2D(128, (3, 3), activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))
# Third convolutional block
model.add(layers.Conv2D(256, (3, 3), activation='relu', 
                        kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Flatten())
# Dense layer
model.add(layers.Dense(512, activation='relu', 
                       kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10)) 

# Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Plotting training and validation accuracy/loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
