import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        # Model architecture with CNN layers and fully connected layers
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 128),  # Adjusted for CNN output
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# Hyperparameters
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "dropout_rate": 0.2
}

# Initialize wandb
wandb.init(project="mnist-mlops", config=config)

# Data transformation and loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Initialize model, loss, and optimizer
model = LogisticRegression(input_dim=32 * 14 * 14, output_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

# Add model monitoring
wandb.watch(model, log="all")

# Log sample misclassified images
def log_misclassifications(model, test_loader):
    misclassified_images = []
    model.eval()
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        for i in range(len(pred)):
            if pred[i] != target[i]:
                misclassified_images.append(wandb.Image(data[i].cpu().numpy(), caption=f"Pred: {pred[i]}, Actual: {target[i]}"))
    wandb.log({"misclassified_images": misclassified_images})

# Log sample predictions
def log_predictions(model, data, target, num_samples=10):
    model.eval()
    images = []
    for i in range(num_samples):
        img = data[i].unsqueeze(0)
        label = target[i].item()
        
        with torch.no_grad():
            output = model(img)
            pred = output.argmax(dim=1).item()
        
        images.append(wandb.Image(data[i], caption=f"Pred: {pred}, Label: {label}"))
    wandb.log({"predictions": images})

# Training loop with detailed logging
for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

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

    wandb.log({"epoch": epoch, "avg_training_loss": total_loss / len(train_loader)})

    # Validation
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.tolist())
            all_labels.extend(target.tolist())

    accuracy = correct / len(test_loader.dataset)
    precision = sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_preds)
    recall = sum([l == p for l, p in zip(all_labels, all_preds)]) / len(all_labels)

    wandb.log({
        "epoch": epoch,
        "validation_accuracy": accuracy,
        "validation_precision": precision,
        "validation_recall": recall
    })

    # Log predictions for visualization
    log_predictions(model, next(iter(test_loader))[0][:10], next(iter(test_loader))[1][:10])
    
    # Save the model after training
    torch.save(model.state_dict(), "mnist_model_ver2.pth")

wandb.finish()
