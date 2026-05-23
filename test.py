import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import random

IMG_SIZE = 128
BATCH_SIZE = 16


def display_all_images(images, true_labels, pred_labels, class_names, cols=4):
    """Отображает все изображения в сетке"""
    num_images = len(images)
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(4 * cols, 4 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)

        img = images[i]
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)

        plt.imshow(img)

        true_class = class_names[true_labels[i]]
        pred_class = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'

        plt.title(f'True: {true_class}\nPred: {pred_class}', color=color, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Datasets/Test",
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names
print("Классы:", class_names)

model = load_model("Models/cnn_model2.h5")

loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_true = []
y_pred = []
all_images = []
all_labels = []

for images, labels in test_dataset:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    all_images.extend(images.numpy())
    all_labels.extend(labels.numpy())

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


all_images = np.array(all_images)
all_labels = np.array(all_labels)
y_pred = np.array(y_pred)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

fp = cm[0][1]  # False Positive
fn = cm[1][0]  # False Negative

plt.figure(figsize=(5, 4))
plt.bar(
    ["False Positive", "False Negative"],
    [fp, fn]
)

plt.title("Errors Type I and II")
plt.ylabel("Count")
plt.show()
display_all_images(all_images, all_labels, y_pred, class_names, cols=5)