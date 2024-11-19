import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

def deprocess_image(img):
    img = img[0]
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.clip(img, 0, 255).astype('uint8')
    return img

def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    a = tf.reshape(tensor, [-1, channels])
    n = tf.shape(a)[0]
    return tf.matmul(a, a, transpose_a=True) / tf.cast(n, tf.float32)

def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def compute_loss(outputs, content_features, style_features):
    style_outputs, content_output = outputs[:-1], outputs[-1]
    style_loss_total = tf.add_n(
        [style_loss(style_features[i], style_outputs[i]) for i in range(len(style_layers))]
    )
    style_loss_total *= style_weight / len(style_layers)

    content_loss_total = content_loss(content_features, content_output) * content_weight

    return style_loss_total + content_loss_total

def style_loss(style, generated):
    S = gram_matrix(style)
    G = gram_matrix(generated)
    return tf.reduce_mean(tf.square(S - G))

content_image_path = "seattle.jpeg"
style_image_path = "butterfly.jpeg"

content_image = preprocess_image(content_image_path)
style_image = preprocess_image(style_image_path)

# --- Load the Pre-trained Model ---
model = vgg19.VGG19(weights="imagenet", include_top=False)
content_layer = 'block5_conv2'
style_layers = [
    'block1_conv1', 'block2_conv1',
    'block3_conv1', 'block4_conv1', 'block5_conv1'
]

outputs = [model.get_layer(name).output for name in style_layers + [content_layer]]
model = Model(inputs=model.input, outputs=outputs)

# --- Compute Loss ---
style_weight = 1e-6
content_weight = 1e-4

# Extract content and style features
content_outputs = model(content_image)
style_outputs = model(style_image)
content_features = content_outputs[-1]
style_features = [style_layer for style_layer in style_outputs[:-1]]

# --- Optimization Loop ---
generated_image = tf.Variable(content_image, dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=0.02)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        outputs = model(generated_image)
        loss = compute_loss(outputs, content_features, style_features)
    grads = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(grads, generated_image)])
    return loss

# --- Training ---
epochs = 10
steps_per_epoch = 100

for epoch in range(epochs):
    for step in range(steps_per_epoch):
        loss = train_step()
        print(f"Epoch {epoch + 1}, Step {step + 1}, Loss: {loss:.4f}")
    
    img = deprocess_image(generated_image.numpy())
    plt.imshow(img)
    plt.title(f"Epoch {epoch + 1}")
    plt.axis("off")
    plt.show()

final_image = deprocess_image(generated_image.numpy())
plt.imshow(final_image)
plt.title("Final Styled Image")
plt.axis("off")
plt.show()
