import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ======================================================
# Configuration
# ======================================================

DATASET_DIR = "dataset_split"

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")

MODEL_DIR = "weights"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 4

EPOCHS = 30

LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)

# ======================================================
# Data Augmentation
# ======================================================

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ======================================================
# Dataset
# ======================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

num_classes = len(train_dataset.classes)

print("\nClasses\n")
for i,c in enumerate(train_dataset.classes):
    print(i, c)

print("\nTotal Classes:", num_classes)

# ======================================================
# Model
# ======================================================

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    num_classes
)

model = model.to(DEVICE)

# ======================================================
# Loss
# ======================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)

# ======================================================
# Training
# ======================================================

best_accuracy = 0

since = time.time()

for epoch in range(EPOCHS):

    print("\nEpoch {}/{}".format(epoch+1,EPOCHS))
    print("-"*40)

    ##########################
    # TRAIN
    ##########################

    model.train()

    running_loss = 0
    running_correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        _,predicted = outputs.max(1)

        total += labels.size(0)

        running_correct += predicted.eq(labels).sum().item()

        running_loss += loss.item()

    train_acc = 100*running_correct/total

    ##########################
    # VALIDATION
    ##########################

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _,predicted = outputs.max(1)

            val_total += labels.size(0)

            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100*val_correct/val_total

    scheduler.step()

    print(f"Train Loss : {running_loss/len(train_loader):.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print(f"Val Acc    : {val_acc:.2f}%")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save({

            "model_state_dict":model.state_dict(),

            "classes":train_dataset.classes,

            "accuracy":best_accuracy

        },MODEL_PATH)

        print("Best model saved.")

elapsed = time.time()-since

print("\nTraining Complete")
print("Best Validation Accuracy : {:.2f}%".format(best_accuracy))
print("Training Time : {:.2f} minutes".format(elapsed/60))
print("Saved to :",MODEL_PATH)
