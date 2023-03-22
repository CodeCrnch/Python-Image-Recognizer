import tensorflow as tf
from tensorflow.keras import layers
import os


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    ".",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    validation_split=0.2,
    subset="training",
    seed=42,
    class_names=["car", "not_car"]
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    ".",
    image_size=(224, 224),
    batch_size=16,
    label_mode="categorical",
    validation_split=0.2,
    subset="validation",
    seed=42,
    class_names=["car", "not_car"]
)


base_model = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset,
    epochs=10,
    validation_data=validation_dataset,
)

if not os.path.exists('my_model'):
    os.mkdir('my_model')

model.save('my_model/my_model.h5')
