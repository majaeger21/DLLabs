import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

dream_layers = ['mixed3', 'mixed5']
layer_outputs = [base_model.get_layer(name).output for name in dream_layers]

dream_model = tf.keras.Model(inputs=base_model.input, outputs=layer_outputs)

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img
def deprocess_image(img):
    img = img[0]
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img)) * 255  # Normalize to [0, 255]
    return tf.cast(img, tf.uint8)

@tf.function
def gradient_ascent_step(image, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        features = dream_model(image)
        loss = tf.reduce_sum([tf.reduce_mean(f) for f in features])
    gradients = tape.gradient(loss, image)
    gradients = gradients / (tf.math.reduce_std(gradients) + 1e-8)
    image += learning_rate * gradients
    return image

def deepdream(image_path, steps=100, learning_rate=0.01):
    image = preprocess_image(image_path)
    for step in range(steps):
        image = gradient_ascent_step(image, learning_rate)
    return deprocess_image(image)

input_image_path = 'butterfly.jpeg'
dreamed_image = deepdream(input_image_path)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
original_image = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(224, 224))
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("DeepDream Image")
plt.imshow(dreamed_image)
plt.axis('off')
plt.show()
