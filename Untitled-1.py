
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ==========================================
# 1. Reproducibility and device
# ==========================================

torch.manual_seed(42)

device = torch.device("cpu")
print("Device:", device)


# ==========================================
# 2. Load and prepare the data
# ==========================================

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ==========================================
# 3. Visualize sample images
# ==========================================

images, labels = next(iter(train_loader))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")
    ax.set_title(f"Label: {labels[i].item()}")
    ax.axis("off")

plt.suptitle("MNIST Handwritten Digits")
plt.tight_layout()
plt.show()


# ==========================================
# 4. Build the neural network
# ==========================================

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.network = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)


model = MNISTModel().to(device)


# ==========================================
# 5. Loss function and optimizer
# ==========================================

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ==========================================
# 6. Training and evaluation functions
# ==========================================

def run_epoch(loader, training):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = loss_function(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        if not training:
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.tolist())

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy, all_predictions, all_labels


# ==========================================
# 7. Train the model
# ==========================================

epochs = 5

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    train_loss, train_acc, _, _ = run_epoch(
        train_loader, training=True
    )

    val_loss, val_acc, _, _ = run_epoch(
        val_loader, training=False
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.2%} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_acc:.2%}"
    )


# ==========================================
# 8. Plot loss and accuracy
# ==========================================

epochs_range = range(1, epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, train_losses, marker="o", label="Train Loss")
plt.plot(epochs_range, val_losses, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)
plt.plot(
    epochs_range,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_curve.png", dpi=150)
plt.show()


# ==========================================
# 9. Final evaluation on test data
# ==========================================

test_loss, test_acc, predictions, true_labels = run_epoch(
    test_loader, training=False
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2%}")


# ==========================================
# 10. Confusion matrix
# ==========================================

matrix = confusion_matrix(
    true_labels,
    predictions,
    labels=list(range(10))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=list(range(10))
)

display.plot(cmap="Blues", values_format="d")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()


# ==========================================
# 11. Show sample predictions
# ==========================================

model.eval()

sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images.to(device)

with torch.no_grad():
    sample_outputs = model(sample_images)
    sample_predictions = sample_outputs.argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(sample_images[i].cpu().squeeze(), cmap="gray")

    predicted = sample_predictions[i].item()
    actual = sample_labels[i].item()

    ax.set_title(
        f"Pred: {predicted} | True: {actual}",
        color="green" if predicted == actual else "red"
    )
    ax.axis("off")

plt.suptitle("Model Predictions: Green = Correct, Red = Incorrect")
plt.tight_layout()
plt.savefig("sample_predictions.png", dpi=150)
plt.show()


# ==========================================
# 12. Save the trained model
# ==========================================

torch.save(model.state_dict(), "mnist_model.pth")

print("Model saved to mnist_model.pth")
print("Project completed successfully!")
