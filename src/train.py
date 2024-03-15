import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import config
import utils
import torchvision.models as models
from model import ObjectDetector
torch.backends.cudnn.benchmark = True

def main():
    model = ObjectDetector().to(config.DEVICE)
    optimizer = Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed

    train_loader, test_loader, _ = utils.get_loaders("../BMW_dataset/train.csv", "../BMW_dataset/test.csv")

    # Training loop
    num_epochs = 10  # Adjust number of epochs
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.permute(0, 3, 1, 2).to(config.DEVICE).type(torch.cuda.FloatTensor)  # BUG fixed
            labels_t = torch.tensor(labels)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels_t)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training progress (optional)
        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Save the trained model (optional)
    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()