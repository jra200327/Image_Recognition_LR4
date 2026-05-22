import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

IMG_SIZE = 128
BATCH_SIZE = 16

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

for images, labels in test_dataset:
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

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