import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_Dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_Dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_Loader = DataLoader(train_Dataset, batch_size=BATCH_SIZE, shuffle=True)
test_Loader = DataLoader(test_Dataset, batch_size=BATCH_SIZE, shuffle=False)


class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10) 
        self.relu3 = nn.ReLU()
        self.output_layer = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.output_layer(x)
        return x


def unnormalize(tensor):
    return tensor * 0.5 + 0.5 


model = DigitClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ---- training loop ----
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for images, labels in train_Loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_Loader)

    # ---- evaluate on the test set after each epoch ----
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_Loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = (total_correct / total_samples) * 100
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_train_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

print("\nTraining completed successfully!")

# ---- visualize a batch of predictions ----
model.eval()
dataiter = iter(test_Loader)
images, labels = next(dataiter)
images_device = images.to(device)

with torch.no_grad():
    outputs = model(images_device)
    _, predicted_classes = torch.max(outputs, 1)

predicted_classes = predicted_classes.cpu()

fig = plt.figure(figsize=(12, 6))
for idx in range(10):
    ax = fig.add_subplot(2, 5, idx + 1, xticks=[], yticks=[])
    image_to_show = unnormalize(images[idx]).squeeze()
    ax.imshow(image_to_show, cmap="gray")

    true_label = labels[idx].item()
    predicted_label = predicted_classes[idx].item()

    color = "green" if predicted_label == true_label else "red"
    ax.set_title(f"True: {true_label}\nPred: {predicted_label}", color=color)

plt.tight_layout()
plt.savefig("predictions.png")
plt.show()