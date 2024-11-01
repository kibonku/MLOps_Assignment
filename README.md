
# FastAPI MNIST Prediction Service

This project is an end-to-end machine learning pipeline for MNIST digit prediction using a Logistic Regression model built with PyTorch. The model is served through FastAPI, containerized with Docker, and monitored with Weights and Biases (wandb) for training performance.

## Project Overview

Key components of this project:
- **Model Training**: Trains a Logistic Regression model on the MNIST dataset using PyTorch.
- **Model Monitoring**: Logs model performance to Weights and Biases (wandb) for tracking.
- **FastAPI Service**: Provides an HTTP API for digit prediction from handwritten images.
- **Docker Containerization**: Containerizes the FastAPI application for easy deployment.

## Instructions for Running the Code Locally

### Prerequisites

- **Python 3.7+**
- **pip** package manager
- **Weights and Biases** account (optional)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
```

### 2. Install Dependencies

Create a virtual environment and install dependencies (recommended).

```bash
# Create virtual environment
python -m venv env
source env/bin/activate  # Use `env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the FastAPI Application

To run the FastAPI application locally, use:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` in your browser to view API documentation and test the endpoints.

### 4. Train the Model (Optional)

If you need to retrain the model, run the training script:

```bash
python train.py
```

This will log training performance to Weights and Biases (wandb).

## Instructions for Building and Running the Docker Container

### 1. Build the Docker Image

Ensure Docker is installed and running, then build the Docker image from the project’s root directory.

```bash
docker build -t fastapi-mnist-app .
```

### 2. Run the Docker Container

Run the Docker container, mapping port 8000 on your local machine to port 8000 in the container.

```bash
docker run -p 8000:8000 fastapi-mnist-app
```

Once the container is running, the FastAPI application is accessible at `http://localhost:8000`.

### 3. Test the API

Go to `http://localhost:8000/docs` in your browser to test the `/predict` endpoint and upload an image for prediction.

## Weights and Biases (wandb) Report

Model training and performance tracking are logged in Weights and Biases (wandb). View the full report at the link below:

[wandb Report URL](https://wandb.ai/yourusername/yourprojectname)  <!-- Replace with the actual wandb report URL -->

## Project Structure

```
project-folder/
├── Dockerfile                # Dockerfile for containerizing the application
├── main.py                   # FastAPI application code
├── train.py                  # Model training code
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── other-files               # Other supporting files
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

If you have questions or encounter issues, please open an issue in the GitHub repository.
