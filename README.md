
# Cardiac Conduction Simulation: A Hybrid Deep Learning Approach

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%7C%20Keras-orange.svg)](https://www.tensorflow.org/)

A web-based platform that uses a hybrid deep learning model to classify realistic ECG signals, helping in the early and accurate detection of cardiac abnormalities like Myocardial Infarction and arrhythmia.

---

## 📖 Table of Contents
* [Overview](#-overview)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Technology Stack](#-technology-stack)
* [Methodology](#-methodology)
* [Results](#-results)
* [Setup and Installation](#-setup-and-installation)
* [Future Enhancements](#-future-enhancements)

---

## 🔭 Overview

Cardiovascular diseases (CVDs) are a leading cause of mortality worldwide. The electrocardiogram (ECG) is a crucial diagnostic tool, but manual analysis is often time-consuming, subjective, and prone to human error. This project addresses these challenges by developing an automated, accurate, and efficient system for ECG image classification.

We implement and compare four sophisticated deep learning models: **CNN, MobileNet, DenseNet, and a novel hybrid MobileNet+LSTM model**. The system classifies ECG images into four distinct categories:
1.  **Normal Heartbeat**
2.  **Myocardial Infarction (MI)**
3.  **History of MI**
4.  **Abnormal Heartbeat (Arrhythmia)**

The hybrid MobileNet+LSTM model, which leverages MobileNet for spatial feature extraction and LSTM for temporal sequence analysis, demonstrated superior performance, highlighting the potential of hybrid architectures in medical diagnostics.

## ✨ Key Features

- **Automated ECG Classification**: Upload an ECG image and get an instant classification.
- **Hybrid Deep Learning Model**: Combines the power of CNNs (MobileNet) and RNNs (LSTM) to capture both spatial and temporal features of ECG signals.
- **Comparative Analysis**: Provides a detailed comparison of multiple state-of-the-art deep learning architectures.
- **User-Friendly Web Interface**: A clean and simple web application built with Flask for easy interaction.
- **Secure User Authentication**: System includes user registration and login functionality.



## 🏗️ System Architecture

The system is designed with a clear separation between the user, the web application, and the backend machine learning model.
## 🛠️ Technology Stack

This project is built with a modern and robust technology stack:

- **Backend**: `Python`, `Flask`
- **Deep Learning**: `TensorFlow`, `Keras`
- **Data Processing**: `Pandas`, `NumPy`, `Scikit-learn`
- **Frontend**: `HTML`, `CSS`, `JavaScript`
- **Database**: `MySQL` (managed with SQLYog)

## 🔬 Methodology

The project follows a structured machine learning workflow:

1.  **Data Collection**: The dataset was sourced from Kaggle's ECG Image Collection.
2.  **Preprocessing**: Images were resized to a uniform `224x224` dimension and normalized. Data augmentation techniques were applied to enhance model robustness.
3.  **Feature Extraction**: A pre-trained **MobileNet** model (on ImageNet) was used as a static feature extractor to generate meaningful embeddings from the ECG images.
4.  **Model Training**: Four different models were trained on the extracted features:
    - A custom **Convolutional Neural Network (CNN)**
    - A fine-tuned **MobileNet** classifier
    - A fine-tuned **DenseNet** classifier
    - A hybrid **MobileNet + LSTM** model
5.  **Evaluation**: Models were evaluated on key metrics including Accuracy, Precision, Recall, and F1-Score to determine the most effective architecture.

## 📊 Results

The comparative analysis revealed that the **hybrid MobileNet + LSTM model** outperformed the other models.

- **MobileNet (Validation Accuracy): ~89.78%**
- **DenseNet (Validation Accuracy): ~84.41%**
- **CNN (Validation Accuracy): ~87.63%**

The superior performance of the hybrid model is attributed to its ability to process both the spatial patterns in the ECG image (via MobileNet) and the temporal dependencies in the cardiac cycle (via LSTM). This dual-pronged approach is highly effective for classifying complex bio-signals.

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(Create a `requirements.txt` file in your repository)*
    ```bash
    pip install -r requirements.txt
    ```
    A typical `requirements.txt` would look like this:
    ```
    Flask
    tensorflow
    pandas
    numpy
    scikit-learn
    mysql-connector-python
    # Add other libraries you used
    ```

4.  **Set up the database:**
    - Create a MySQL database.
    - Import the necessary schema or tables.
    - Update the database connection details in your Flask application file (e.g., `app.py`).

5.  **Run the application:**
    ```bash
    python app.py
    ```

6.  **Open your browser** and navigate to `http://127.0.0.1:5000`.

## 🔮 Future Enhancements

- **Integration of Multi-Lead ECG Data**: Expand the model to analyze data from multiple ECG leads for more comprehensive and reliable diagnostics.
- **Real-Time Analysis**: Develop capabilities for real-time ECG stream processing from wearable devices.
- **Model Explainability**: Incorporate techniques like LIME or SHAP to provide visual explanations for model predictions, increasing trust and interpretability for clinicians.
- **Deployment to Cloud**: Package the application in a Docker container and deploy it to a cloud platform like AWS, Azure, or Heroku for wider accessibility.


