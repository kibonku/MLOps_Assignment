import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # FIXME: Implement the model architecture
         # Hint: Consider adding hidden layers and dropout
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # FIXME: Implement the forward pass
        return self.network(x)

# Initialize wandb
config = {
    "learning_rate": 0.01,
    "epochs": 10,
    "batch_size": 64,
    "hidden_layers": [128, 64],
    "dropout_rate": 0.2
}

# Initialize wandb
wandb.init(project="mnist-mlops", config=config)

# Load and preprocess data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize model, loss, and optimizer
model = LogisticRegression(input_dim=28*28, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Add model monitoring
wandb.watch(model, log="all")

# Define function to log sample predictions
def log_predictions(model, data, target, num_samples=10):
    model.eval()
    images, predictions, labels = [], [], []

    # Only log a specified number of samples
    for i in range(num_samples):
        img = data[i].view(-1, 28 * 28)
        label = target[i].item()
        
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
        
        # Append data for wandb logging
        images.append(wandb.Image(data[i], caption=f"Pred: {pred}, Label: {label}"))
        predictions.append(pred)
        labels.append(label)

    # Log images and predictions to wandb
    wandb.log({"predictions": images})

# Training loop
# In the training loop, add more detailed logging
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)  # Flatten MNIST images
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # FIXME: Log additional metrics (e.g., learning rate, gradient norms)
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        wandb.log({
            "epoch": epoch,
            "batch_idx": batch_idx,
            "training_loss": loss.item(),
            "learning_rate": config["learning_rate"],
            "gradient_norm": grad_norm
        })

        optimizer.step()
        total_loss += loss.item()

    # Log training loss
    wandb.log({"epoch": epoch, "avg_training_loss": total_loss / len(train_loader)})

    # Validation
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28*28)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.tolist())
            all_labels.extend(target.tolist())

    accuracy = correct / len(test_loader.dataset)

    # After validation, log more metrics
    # FIXME: Log additional validation metrics (e.g., precision, recall)
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    precision = (all_preds[all_preds == all_labels] == all_labels[all_preds == all_labels]).sum().item() / len(all_preds)
    recall = (all_labels[all_labels == all_preds] == all_preds[all_labels == all_preds]).sum().item() / len(all_labels)

    # Log validation metrics
    wandb.log({
        "epoch": epoch,
        "validation_accuracy": accuracy,
        "validation_precision": precision,
        "validation_recall": recall
    })

    # Call this function at the end of each epoch or at the end of training
    log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])
    
    # After training, save the model
    torch.save(model.state_dict(), "mnist_model.pth")

wandb.finish()
