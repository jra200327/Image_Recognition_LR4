import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 128
BATCH_SIZE = 16

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Datasets/Train",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Datasets/Test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print(class_names)

# только лёгкая аугментация
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
])

model = models.Sequential([
    layers.Rescaling(1./255),
    data_augmentation,

    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# early stopping (очень важно)
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=50,
    callbacks=[callback]
)

model.save("Models/cnn_model2.h5")

print("Model saved to Models/cnn_model2.h5")