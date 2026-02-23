import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224

# Load trained model
model = tf.keras.models.load_model("traffic_classifier_resnet.h5")

# Recreate training generator (to get correct class order)
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    "cnn_dataset/train",   # make sure this path is correct
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

class_names = list(train_gen.class_indices.keys())

# Load test image
image_path = "stop3.jpg"   # put your stop image here

img = cv2.imread(image_path)

if img is None:
    print("Image not found! Check file name.")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
pred = model.predict(img)[0]

print("\n--- ALL CLASS PROBABILITIES ---\n")
for i, prob in enumerate(pred):
    print(f"{class_names[i]} : {round(float(prob),4)}")

class_id = np.argmax(pred)

print("\nFinal Prediction:", class_names[class_id])
print("Confidence:", float(pred[class_id]))