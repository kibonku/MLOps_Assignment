from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = FastAPI()

# Define the model architecture
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Load the trained model
model_path = "mnist_model.pth"  # Make sure to specify the correct path
model = LogisticRegression(input_dim=28*28, output_dim=10)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure single-channel grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST dataset mean and std
])

# Define the predict endpoint with input validation and error handling
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Error handling for invalid file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    # Load and preprocess the image
    try:
        image = Image.open(io.BytesIO(await file.read())).convert('L')
        image = preprocess(image).unsqueeze(0)
    except Exception as e:
        return {"error": str(e)}

    # Model inference
    with torch.no_grad():
        output = model(image.view(-1, 28 * 28))  # Flatten the image to match input layer size
        prediction = output.argmax(dim=1).item()

    # Return the prediction as JSON
    return {"prediction": prediction}

# Run the application locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
