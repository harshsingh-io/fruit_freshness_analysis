import nbformat as nbf

# Create a new notebook object
nb = nbf.v4.new_notebook()

# Create cells with content from the Python script, split for clarity
cells = [
    nbf.v4.new_code_cell("""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    """),
    nbf.v4.new_code_cell("""
# Define paths to the training and test data
train_data_path = r'D:\\harsh\\archive2\\dataset\\Train'
test_data_path = r'D:\\harsh\\archive2\\dataset\\Test'
    """),
    nbf.v4.new_code_cell("""
# Data augmentation with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2  # Set aside 20% for validation
)

# Training and validation generators
train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
    """),
    nbf.v4.new_code_cell("""
# Test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only rescale for test data
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important to keep the order for evaluation
)
    """),
    nbf.v4.new_code_cell("""
# Print class indices to confirm number of classes
print("Class indices: ", train_generator.class_indices)
    """),
    nbf.v4.new_code_cell("""
# Load VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze the base model
base_model.trainable = False

# Create the hybrid model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])
    """),
    nbf.v4.new_code_cell("""
# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    """),
    nbf.v4.new_code_cell("""
# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
    """),
    nbf.v4.new_code_cell("""
# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, reduce_lr]
)
    """),
    nbf.v4.new_code_cell("""
# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy:.2f}')
    """),
    nbf.v4.new_code_cell("""
# Optional: Fine-tuning the model
# Unfreeze the last few layers of the base model
base_model.trainable = True
for layer in base_model.layers[:-4]:  # Unfreeze the last 4 layers
    layer.trainable = False

# Recompile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training for additional epochs
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=15,  # Fewer epochs for fine-tuning
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, reduce_lr]
)

# Final evaluation on the test set after fine-tuning
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Final Test Accuracy after fine-tuning: {test_accuracy:.2f}')
    """)
]

# Add cells to the notebook
nb['cells'] = cells

# Save the notebook
notebook_path = "/mnt/data/Converted_Python_Script_to_Jupyter_Notebook.ipynb"
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

notebook_path
