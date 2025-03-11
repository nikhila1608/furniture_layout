import numpy as np
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
current_dir = os.getcwd()  
file_path = os.path.join(current_dir, 'dataset.json')

with open(file_path, "r") as f:
    dataset = json.load(f)

# Step 2: Convert dataset to NumPy arrays
X = []  # Input: Room dimensions + furniture properties
y = []  # Output: Optimized furniture positions

# Find the maximum number of furniture items in any room
max_furniture_items = max(len(data["furniture"]) for data in dataset)

for data in dataset:
    room = [data["room"][0], data["room"][1]]  # Room width, height
    furniture = []
    
    for item in data["furniture"]:
        furniture.extend([item["w"], item["h"], item["x"], item["y"]])  
    
    # Pad with zeros to ensure equal length
    furniture.extend([0, 0, 0, 0] * (max_furniture_items - len(data["furniture"]))) 

    X.append(room + furniture)

    # Target positions (x, y of furniture)
    y_data = []
    for item in data["furniture"]:
        y_data.extend([item["x"], item["y"]])
    
    # Pad with zeros for consistency
    y_data.extend([0, 0] * (max_furniture_items - len(data["furniture"])))
    y.append(y_data)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Step 3: Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define the Neural Network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(y.shape[1])  # Output layer to predict furniture positions
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 5: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Save the trained model
model.save("furniture_layout_model.h5")
print("Model training complete. Saved as 'furniture_layout_model.h5'.")
