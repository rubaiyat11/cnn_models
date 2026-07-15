import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet18_Weights.DEFAULT      #Taking resnet default weights

model = resnet18(weights=weights).to(device)  #Creating Instance of resnet model with default weights

for param in model.parameters():        #Freeze all parameters
    param.requires_grad = False     

for param in model.layer4.parameters():     #Unfreeze last layer parameters
    param.requires_grad = True

model.fc = nn.Linear(512, 10).to(device)    #Create a new NN layer for training CIFAR dataset
model.fc.requires_grad_(True)

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
    list(model.layer4.parameters()) +
    list(model.fc.parameters()),
    lr=0.0001
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

epochs = 12

#Training LOOP
for epoch in range(epochs):     

    model.train()      #Model training state

    running_loss = 0       #Variable for calculating total loss after the loop

    for images, labels in train_loader: #Looping through data in the dataset

        #Put images, labels into GPU
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

    scheduler.step()

    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")


model.eval()

classes = [         #Output classes
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

with torch.no_grad():   #Testing Phase

    for images, labels in test_loader:  #Testing LOOP

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
        
