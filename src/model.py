import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Net(nn.Module):
    def __init__(self):
        super(Connect4Net, self).__init__()

        # Convolutional layers
        # Input: 1 channel (the board), Output: 64 features
        # Kernel: 4x4 is chosen because a win is 4 in a row.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Second conv layer to capture more complex patterns
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)


        # Fully connected layers (The Decision Maker)
        # We need to calculate the input size after flattening.
        # For a 6x7 board and the above conv layers, it comes to 128 * 6 * 5 = 3840
        self.fc1 = nn.Linear(3840, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)  # Output: Single value for win/loss/draw prediction
    
    def forward(self, x):
        # X shape: (batch_size, 1, 6, 7)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Flatten: (batch_size, 128 * 6 * 7) -> (batch_size, 5376)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))  # Tanh forces output between -1 and 1

        return x
    
if __name__ == "__main__":
    model = Connect4Net()
    dummy_input = torch.randn(1, 1, 6, 7)  # Batch size of 1
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (1, 1)
    print("Output value:", output.item())  # Single value between -1 and 1