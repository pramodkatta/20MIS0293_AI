---

# Trademarkia Classifier

A machine learning project that classifies trademark classes based on goods and services entered by users. The model is built using PyTorch, utilizes a BERT-based architecture, and is served through a REST API. This project also includes Docker support and integrates with Weights & Biases (W&B) for tracking experiments.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Model Training](#model-training)
- [REST API](#rest-api)
- [Docker Support](#docker-support)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project focuses on building a machine learning model that predicts the trademark class for a given goods or service description. It leverages a BERT-based transformer model to classify trademark descriptions into 45 different categories based on the **Nice Classification System**. 

### Key Features:
- BERT-based model using PyTorch.
- REST API for serving the model predictions.
- Docker support for containerization.
- W&B integration for tracking training experiments and hyperparameters.

## Project Structure

```plaintext
trademarkia-classifier/
├── .gitignore
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── venv/                     # Python virtual environment
├── src/                      # Source code
│   ├── train.py              # Script to train the model
│   ├── model.py              # Model architecture and dataset
│   ├── data_preprocessing.py # Data preprocessing script
│   ├── api/                  # API-related files
├── data/                     # Data folder
│   ├── idmanual.json         # Trademark data (input dataset)
│   ├── preprocessed_data.csv # Preprocessed data file
└── model/                    # Saved model folder
```

## Setup

### Prerequisites

- Python 3.8 or later
- Docker (optional for containerization)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pramodkatta/20MIS0293_AI.git
   cd trademarkia-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory (if required).
   - Example `.env`:
     ```plaintext
     DEBUG=1
     ```

### Running the Project

1. **Train the model:**
   Run the following command to preprocess the data and train the model:
   ```bash
   python src/train.py
   ```

2. **Run the REST API:**
   To serve the model predictions through an API, run the following:
   ```bash
   python src/api/app.py
   ```

   The API will be accessible at `http://localhost:5000/`.

## Model Training

To train the model, ensure that your dataset (`idmanual.json`) is present in the `data/` directory. The training process uses a BERT-based classifier that is implemented in `src/model.py`.

Run the training script as follows:

```bash
python src/train.py
```

You can monitor the training process on Weights & Biases (W&B) if integrated.

## REST API

The project includes a simple REST API to classify trademark descriptions using the trained model.

### API Endpoints

- `POST /predict`: Classify a goods or service description.
  - **Request Body**:
    ```json
    {
      "description": "Your goods or service description here"
    }
    ```
  - **Response**:
    ```json
    {
      "class": "Predicted class for the given description"
    }
    ```

### Running the API Locally

```bash
python src/api/app.py
```

### Example Usage (cURL):

```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"description": "Bank note acceptors for separating good bank notes from counterfeits"}'
```

## Docker Support

This project includes a `Dockerfile` and `docker-compose.yml` for containerized deployment.

### Build and Run the Project with Docker

1. **Build the Docker image:**
   ```bash
   docker-compose up --build
   ```

2. **Run the container:**
   ```bash
   docker-compose up
   ```

The API will be available at `http://localhost:8000/`.

---

