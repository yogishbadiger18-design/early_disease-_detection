import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths
train_dir = 'train'
test_dir = 'test'
img_size = (150, 150)
batch_size = 32
model_file = 'disease_model.h5'
accuracy_plot_file = 'accuracy.png'
loss_plot_file = 'loss.png'
classification_report_file = 'classification_report.png'

# 1️⃣ Data Generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   zoom_range=0.2,
                                   shear_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir,
                                               target_size=img_size,
                                               batch_size=batch_size,
                                               class_mode='categorical')
test_gen = test_datagen.flow_from_directory(test_dir,
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            class_mode='categorical',
                                            shuffle=False)

class_names = list(train_gen.class_indices.keys())

# 2️⃣ Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=img_size+(3,)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3️⃣ Train
history = model.fit(train_gen,
                    validation_data=test_gen,
                    epochs=20)

# 4️⃣ Save Model
model.save(model_file)

# 5️⃣ Plot Training Results
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(accuracy_plot_file)

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig(loss_plot_file)

plt.show()

# 6️⃣ Evaluate and Save Confusion Matrix
import numpy as np
test_gen.reset()
Y_pred = model.predict(test_gen)
y_pred = np.argmax(Y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_true = test_gen.classes
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(classification_report_file)
plt.show()
