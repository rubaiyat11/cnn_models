import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights).to(device)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10).to(device)


train_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=weights.transforms()
)

test_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=weights.transforms()
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


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

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
        f"Epoch {epoch+1}",
    )
    print(
        f"Loss: {running_loss/len(train_loader):.4f}"
    )


model.eval()

classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

correct = 0
total = 0

sample_predictions = None
sample_labels = None

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions==labels).sum().item()

        total += len(labels)

        if sample_labels is None:
            sample_predictions = predictions
            sample_labels = labels

    accuracy = correct/total

    print(f"Accuracy: {accuracy*100:.2f}%")

    for i in range(5):

        print(f"Predicted: {classes[sample_predictions[i]]}")
        print(f"Actual:    {classes[sample_labels[i]]}")
        print()
        
