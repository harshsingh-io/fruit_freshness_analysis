
# Freshness Analysis of Fruits Using Deep Learning

This project utilizes a deep learning model to analyze the freshness of fruits from images. The project has gone through multiple versions to improve model accuracy and real-world applicability. The current version (v2) integrates model fine-tuning and real-world use-case predictions, building upon the previous version's (v1) baseline model.
## Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/K7LAQh5VaDM?si=8CCu68YNAl4eJSM1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Documentation

[View Documentation](https://github.com/harshsingh-io/fruit_freshness_analysis/blob/main/Key%20Components_20241020_114930_0000.pdf)

## Project Overview

- **v1**: Initial version with a pre-trained VGG16 model to classify fruit freshness, achieving an accuracy of around 88%.
- **v2**: Enhanced the model through fine-tuning, implemented a prediction function for real-world use cases, and increased the overall flexibility of the model.

## Key Features

- **Deep Learning Model**: Utilizes the VGG16 network as a base model with layers frozen during the initial training phase to leverage pre-trained image features.
- **Fine-Tuning**: Unfreezes the last few layers of the VGG16 base model to adapt the pre-trained network to the specific task of fruit freshness classification, improving the model's accuracy.
- **Data Augmentation**: Implements data augmentation to increase dataset diversity, making the model more robust and generalizable.
- **Practical Application**: The model includes a function to classify new fruit images, making it practically applicable for real-world usage scenarios.

## Technologies Used

- **TensorFlow**
- **Keras**
- **Python**
- **VGG16**

## Project Structure

- **v1**: Initial implementation with basic training and evaluation.
- **v2**: Fine-tuning of the model, added prediction capability for individual fruit images.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/harshsingh-io/fruit_freshness_analysis.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Setup

- The dataset consists of fruit images categorized by freshness.
- **Training and Validation Data**: Training images are divided into training and validation sets using `ImageDataGenerator`.
- **Test Data**: A separate test dataset is used for evaluating model performance.

## Model Training

### v1: Initial Training

- **Model Architecture**: 
  - Uses the pre-trained VGG16 model as the base.
  - Additional layers include `Flatten`, `Dense`, `Dropout`, and an output layer for classification.
- **Training**:
  - The model is trained with the base VGG16 layers frozen, using `Adam` optimizer.
  - **Callbacks**:
    - `EarlyStopping` for monitoring validation loss and avoiding overfitting.
    - `ReduceLROnPlateau` to reduce learning rate when the validation loss stops improving.
- **Performance**: Achieved a test accuracy of **88%**.

### v2: Fine-Tuning and Enhancement

- **Fine-Tuning**:
  - Unfroze the last few layers of the VGG16 base model.
  - Reduced the learning rate to adapt the model to the fruit freshness classification.
- **Training**:
  - Continued training for additional epochs to refine model accuracy.
- **Result**: Improved the robustness of the model and achieved a more accurate representation of freshness levels.

## Prediction Function

A function `classify_image()` is provided to predict the class of a new fruit image.

### Example Usage:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load model and image
model_path = 'trainedModel.h5'
image_path = 'path_to_fruit_image.jpg'

# Function to classify the image
result = classify_image(image_path, model_path)
print("Predicted class:", result)
```

## How to Run

1. **Training**: To train the model, simply execute the training script (`train_model.py`).
2. **Fine-Tuning**: To further improve the model, the fine-tuning section can be run.
3. **Prediction**: Use the `classify_image()` function to predict freshness from new images.

## Save and Load Model

- The trained model is saved as `trainedModel.h5` and can be reloaded for future use:
  ```python
  model = load_model('trainedModel.h5')
  ```

## Results

- **Validation Accuracy**: Achieved around 88% accuracy during initial training.
- **Test Accuracy**: Final test accuracy after fine-tuning was **88%**.
- **Improvement**: Fine-tuning provided a more adaptive model capable of better generalization on unseen data.

## Conclusion

The Freshness Analysis project (v2) builds upon v1 with several optimizations and improvements, including fine-tuning and enhanced data preprocessing. These changes make the model more robust and applicable for real-world scenarios where determining fruit freshness can be of immense value.

## Future Work

- **Data Expansion**: Incorporate more diverse datasets for improved generalization.
- **Deployment**: Develop an API to integrate the model with mobile or web applications.
- **Real-Time Inference**: Implement faster inference mechanisms for real-time applications.


