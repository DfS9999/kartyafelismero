import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
import time
import timm

# args
def parse_args():
    parser = argparse.ArgumentParser(prog='Model training')
    parser.add_argument('train_dir', type=str)
    parser.add_argument('validation_dir', type=str )
    parser.add_argument('test_dir', type=str)
    parser.add_argument("model", type=str, choices=['A', 'B', 'C'])
    return parser.parse_args()

################################################################
# TRANSFORM
################################################################
transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_kaggle_cards = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

################################################################
# DATASET
################################################################
class CardDataset(Dataset):
  def __init__(self, images_dir, transform=None):
    self.data = ImageFolder(root=images_dir, transform=transform)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]

################################################################
# MODEL A
################################################################ 
class CardClassifierModel_A(nn.Module):
    def __init__(self, class_count):
        super(CardClassifierModel_A, self).__init__()
        # -> [3, 224, 224]
        # 1. block 4x4x3x8
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=1)  # -> [8, 221, 221]
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=6, stride=6)                              # -> [8, 36, 36]

        # 2. block 2x2x8x16
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2, stride=1) # -> [16, 35, 35]
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)                              # -> [16, 8, 8]

        self.fc1 = nn.Linear(in_features=16 * 8 * 8, out_features=512)                  # -> 512
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=class_count)                 # -> class_count

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

################################################################
# MODEL B
################################################################ 
class CardClassifierModel_B(nn.Module):
    def __init__(self, class_count):
        super(CardClassifierModel_B, self).__init__()
        # -> [3, 224, 224]
        # 1. block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        # ---> [32, 56, 56]
        # 2. block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        # ---> [64, 14, 14]
        # 3. block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ---> [128, 7, 7]
        # ---> 128*7*7

        self.fc1   = nn.Linear(in_features=128 * 7 * 7, out_features=512) # 6272 -> 512
        self.relu4 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2   = nn.Linear(in_features=512, out_features=class_count) # 512 -> class_count

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x)))) 
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)                          

        x = self.drop1(self.relu4(self.fc1(x)))         
        x = self.fc2(x)                                    

        return x

################################################################
# MODEL C
################################################################ 
class CardClassifierModel_Efficientnet_b0(nn.Module):
    def __init__(self, class_count):
        super(CardClassifierModel_Efficientnet_b0, self).__init__()
        # using pre-trained model 'efficientnet_b0' [3,224,224]
        self.base_model = timm.create_model(model_name='efficientnet_b0', pretrained=True)

        #   - removing default classification layer(1280->1000)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        #   - replace it with new classifier (1280->class_count)
        b0_model_default_out_size = self.base_model.classifier.in_features
        self.classifier = nn.Linear(in_features=b0_model_default_out_size,
                                    out_features=class_count)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################
# training the model
################################################################
def train(args):
    # get count of outputs
    class_count = len(ImageFolder(args.train_dir).class_to_idx.items())
    print(f"Train dataset has {class_count} classes.")
    
    # select transform and class count
    if (len(class_count) == 32):
        classes = 32
        training_transform = transform_base
        print("Loading basic images transform")
    elif (len(class_count) == 53):
        classes = 53
        training_transform = transform_kaggle_cards
        print("Loading basic kaggle images' transform")
    else: 
        print("Unknown dataset. Exiting.")
        exit(1)

    # datasets
    train_dataset = CardDataset(images_dir=args.train_dir, transform=training_transform)
    valid_dataset = CardDataset(images_dir=args.validation_dir, transform=transform_base)
    test_dataset  = CardDataset(images_dir=args.test_dir, transform=transform_base)
    
    # dataloaders
    batch = 64
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch, shuffle=False)
    test_dataloader  = DataLoader(dataset=test_dataset,  batch_size=batch, shuffle=False)
    print(f"Datasets loaded, batch size={batch}")
    
    # gpu if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
        
    # select model
    if (args.model == 'A'):
        print("Using Model A")
        model = CardClassifierModel_A(class_count=classes)
    elif (args.model == 'B'):
        print("Using Model B")
        model = CardClassifierModel_B(class_count=classes)
    elif (args.model == 'C'):
        print("Using Model C")
        model = CardClassifierModel_Efficientnet_b0(class_count=classes)
        
    model.to(device)
    
    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimizer function
    learning_rate = 0.001
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

    ################################################################
    # training loop
    ################################################################
    print("Starting training")
    epochs = 20
    patience = 2
    best_val_loss = float('inf')
    worse = 0
    train_losses, val_losses = [], []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        
        running_loss = 0.0
        batch_idx = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            batch_start_time = time.time()
            print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            batch_time = time.time() - batch_start_time
            print(f"- Current Loss: {loss.item():.4f}, Batch Time: {batch_time:.2f} seconds")
            batch_idx += 1

        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = F.softmax(model(images), dim=1) # -> softmax is needed
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

        val_loss = running_loss / len(valid_dataloader.dataset)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")
        
        # "early stopping"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            worse = 0
            torch.save(model.state_dict(), "model_state_dict.pth")
            print("Model saved !")
        else:
            worse += 1
            if worse >= patience:
                print(f"Early stopping at Epoch {epoch + 1}")
                break

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} sec")

    ################################################################
    # testing
    ################################################################
    print("Training finished, starting test")
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = F.softmax(model(images), dim=1) # -> softmax is needed
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)

            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = correct / total * 100
    total_test_time = time.time() - start_time

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total test time: {total_test_time:.2f} seconds")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
