import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Step 1: Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# Step 2: Generate Images
def create_scatter_plot(data, species, filename):
    plt.figure()
    plt.scatter(data['sepal_length'], data['sepal_width'], c=data.index, cmap='viridis')
    plt.title(f'Scatter Plot of {species}')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.savefig(filename)
    plt.close()

if not os.path.exists('iris_images'):
    os.makedirs('iris_images')

for species in iris_data['species'].unique():
    create_scatter_plot(iris_data[iris_data['species'] == species], species, f'iris_images/{species}.png')

# Step 3: Prepare Image Data
image_files = [f'iris_images/{species}.png' for species in iris_data['species'].unique()]
labels = [species for species in iris_data['species'].unique() for _ in range(1)]  # 1 image per species

# Load and preprocess images
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_image(filename):
    img = load_img(filename, target_size=(64, 64))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

images = np.array([load_and_preprocess_image(f) for f in image_files])
labels = pd.get_dummies(labels).values  # One-hot encoding for labels

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 5: Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes for Iris
])

# Step 6: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Step 8: Evaluate the Model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Step 9: Visualize Results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
