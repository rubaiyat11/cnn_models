import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


train_dataset = datasets.FashionMNIST(
    root="data",
    train =True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class FashionMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.FC1 = nn.Linear(16*7*7, 64)

        self.FC2 = nn.Linear(64, 10)


    def forward(self, x):

        x = self.conv1(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)
        
        x = self.FC1(x)

        x = self.relu(x)

        x = self.FC2(x)

        return x
    

model = FashionMNISTModel()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = loss_fn(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    
    print(
    f"Epoch {epoch+1}"
    )

    print(
        f"Loss: {running_loss/len(train_loader):.4f}"
    )


torch.save(model.state_dict(), "fashion_mnist_model.pth")

model.eval()

classes = [
'T-shirt',
'Trouser',
'Pullover',
'Dress',
'Coat',
'Sandal',
'Shirt',
'Sneaker',
'Bag',
'Boot'
]

correct = 0
total = 0

sample_predictions = None
sample_labels = None

for images, labels in test_loader:

    outputs = model(images)

    predictions = outputs.argmax(dim=1)

    correct += (predictions == labels).sum().item()

    total += len(labels)

    if sample_predictions is None:
        sample_predictions = predictions
        sample_labels = labels


accuracy = correct / total

print(f"Accuracy: {accuracy*100:.2f}%")

for i in range(5):

    print(f"Predicted: {classes[sample_predictions[i]]}")
    print(f"Actual:    {classes[sample_labels[i]]}")
    print()