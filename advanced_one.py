import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fastDP import PrivacyEngine  # Ensure fastDP is installed and available
import time
import config
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPU(s)")
# 1. Load CIFAR-10 dataset
def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    return trainloader

# 2. Define a simple CNN model for CIFAR-10

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        # Define the layers (same as in previous code)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    
def train_model():
    #include the gradient clipping as part of the differential privacy mechanism. 
    #The PrivacyEngine automatically handles the gradient clipping 
    # as well as other privacy-related tasks such as noise addition to gradients, 
    # based on the configuration you provide during initialization
    
    trainloader = get_data_loader()
    model = AdvancedCNN()
    # Parallelize model across GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    from decimal import Decimal
    config.global_gaussian = Decimal(0)
    config.global_clipping = Decimal(0)
    config.count_gaussian = 0
    config.count_clipping = 0
    config.count_total_gaussian = 0
    config.count_total_clipping = 0
    # 3. Define optimizer and privacy engine
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=256,
        sample_size=len(trainloader.dataset),
        epochs=1,  # Change here to only do one epoch
        target_epsilon=2,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
        clipping_value=1.0
    )
    privacy_engine.attach(optimizer)
    # 4. Training loop
    epochs = 1  # Only one epoch
    start_train = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # define loss function, compute loss
            loss = F.cross_entropy(outputs, labels)

            # gradient of the loss w.r.t model parameters computed
            loss.backward()
            
            # PrivacyEngine Gradient Clipping, Adding Gaussian noise
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    end_train = time.time()
    print(f"Finished Training, time = {end_train - start_train}")
    print(f"time_gaussian: {config.global_gaussian}, time_clipping: {config.global_clipping}")
    print(f"count_gaussian: {config.count_gaussian}, count_clipping: {config.count_clipping}") 
    print(f"count_total_gaussian: {config.count_total_gaussian}, count_total_clipping: {config.count_total_clipping}")
    #clipping_gaussian_time()

def clipping_gaussian_time():

    trainloader = get_data_loader()
    model = AdvancedCNN().to(device)

    # 3. Define optimizer and privacy engine
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=256,
        sample_size=len(trainloader.dataset),
        epochs=1,  # Only one epoch
        target_epsilon=2,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
        clipping_value=1.0
    )
    privacy_engine.attach(optimizer)

    epochs = 1  # Only one epoch
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Initialize dictionary to accumulate gradients for each parameter
        accumulated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
    
            # Zero gradients before the backward pass
            optimizer.zero_grad()

            # Forward pass and compute loss
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()

            # Accumulate gradients across batches
            for name, param in model.named_parameters():
                if param.grad is not None:
                    accumulated_gradients[name] += param.grad

            running_loss += loss.item()

        # After looping through all batches, compute the average gradient
        avg_gradients = {name: accumulated_gradients[name] / len(trainloader) for name in accumulated_gradients}
        print(len(trainloader))
        # Clip averaged gradients and measure time
        start_clip = time.time()
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(name)
                print(param)
                param.grad = avg_gradients[name]  # Replace with averaged gradients
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)  # Clip gradients
        end_clip = time.time()
        print(f"Time for gradient clipping: {end_clip - start_clip} seconds")

        # Add Gaussian noise to the averaged gradients and measure time
        # use torch.normal rather than opendp
        noise_std = 1.6
        grad_num = 0
        total_param_element = 0
        start_noise = time.time()
        for name, param in model.named_parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_std, size=param.grad.shape).to(param.device)
                print(noise)
                print(grad_num)
                param.grad += noise  # Add noise to gradients
                grad_num+=1
                total_param_element = torch.numel(param) + total_param_element
                print(total_param_element)
        end_noise = time.time()
        print(f"Time for adding Gaussian noise: {end_noise - start_noise} seconds")

        # Step to update model parameters
        optimizer.step()

        print(f"[{epoch + 1}] epoch loss: {running_loss / len(trainloader):.3f}")

    print("Finished Training")


if __name__ == '__main__':
    train_model()
    
    
    
model = AdvancedCNN()
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
