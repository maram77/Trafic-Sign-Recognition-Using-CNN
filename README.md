# Traffic Sign Recognition Using CNN

This repository contains a Jupyter notebook for training a Convolutional Neural Network (CNN) using Python and TensorFlow/Keras. The notebook demonstrates how to preprocess data, define a CNN model, train it, and evaluate its performance.

## Features
- Data preprocessing, including loading and augmentation.
- Customizable CNN architecture.
- Training and validation with detailed performance metrics.
- Save and load trained models for future use.

## Prerequisites
Make sure you have the following installed:

- Python (>= 3.7)
- Jupyter Notebook
- TensorFlow (>= 2.x)
- NumPy
- Matplotlib
- scikit-learn

Install the required dependencies with:
```bash
pip install requirements.txt
```

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/maram77/Trafic-Sign-Recognition-Using-CNN.git
   cd traffic-sign-detection-cnn-main
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook CNN-training.ipynb
   ```

3. Follow the steps in the notebook to:
   - Load your dataset.
   - Preprocess the data.
   - Configure and train the CNN model.
   - Evaluate the model's performance.

4. Save the trained model for later use or deployment.

## Customization
- Modify the CNN architecture in the model definition section to fit your specific use case.
- Adjust hyperparameters such as learning rate, batch size, and the number of epochs to optimize performance.
- Use custom datasets by replacing the data loading section with your own data pipeline.

---

### Note
For large datasets, ensure sufficient memory and GPU acceleration (e.g., using Google Colab or a local GPU-enabled setup).
