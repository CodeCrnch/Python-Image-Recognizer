import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import os

model = load_model("my_model/my_model.h5")

image_folder = "test/"

for img_filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_filename)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    prediction = model.predict(x)

    if prediction[0][0] > 0.5:
        print(f"{img_filename} is a car.")
        print(prediction[0][0])
    else:
        print(f"{img_filename} is not a car.")
        print(prediction[0][0])
