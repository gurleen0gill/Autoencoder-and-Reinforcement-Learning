## Importing all the necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim 
from matplotlib import pyplot as plt
from torchvision import transforms
import time
import timeit
from tqdm import tqdm


############################################## Question 2 - Autoencoder ###################################################################

############################## Preprocessing ###############################

x_train = pd.read_csv("trainingpix.csv", header= None)
x_test = pd.read_csv("testingpix.csv", header= None)

x_train_tensordata = torch.tensor(x_train.values, dtype=torch.float32).to(device).reshape(-1, 1, 15, 15)
x_test_tensordata = torch.tensor(x_test.values, dtype=torch.float32).to(device).reshape(-1, 1, 15, 15)

x_train_tensordata = torch.flip(x_train_tensordata, dims = [2])
x_test_tensordata = torch.flip(x_test_tensordata, dims = [2])

pixeldata = data.DataLoader(x_train_tensordata, batch_size= 32, shuffle= True)
pixeldata_test = data.DataLoader(x_test_tensordata, batch_size= 10, shuffle= True)

#Checking whether the system has the gpu or cpu to work on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################## Part 1 ###############################

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Define encoding layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.b2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 20, kernel_size=3, stride=2, padding=1)

        # Define decoding (transposed convolution) layers
        self.iconv1 = nn.ConvTranspose2d(20, 128, kernel_size=3, stride=2, padding=1)
        self.b4 = nn.BatchNorm2d(128)
        self.iconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.iconv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2)

    # Encoder function for encoding the data
    def encoder(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.b2(x)
        x = self.conv3(x)
        return x
    
    # Decoder function for reconstructing the encoded data
    def decoder(self, x):
        x = F.relu(self.iconv1(x))
        x = self.b4(x)
        x = F.relu(self.iconv2(x))
        x = self.iconv3(x)
        return x
    
    # Forward function to define the encoding-decoding pipeline
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define transformations for input data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Instantiating the Autoencoder model
AutoEncoder = Autoencoder().to(device)

############################## Training Model ###############################

# Set up training parameters
epochs = 5000
learning_rate = 0.0001
criterion = nn.MSELoss()  # Define the loss function for reconstruction
optimizer = optim.Adam(AutoEncoder.parameters(), lr=learning_rate, weight_decay=1e-4)

# Initialize a list to store training losses and record the start time
trainlosses = []
start_time = time.time()

# Set the Autoencoder to training mode
AutoEncoder.train()

# Loop through the number of epochs
for epoch in range(epochs):
    batchloss = []  # List to store the loss for each batch in the current epoch
    
    # Loop over each batch in the dataset
    for data in pixeldata:
        input_img = data.to(device)  
        optimizer.zero_grad()
        
        # Perform forward pass to get output images and calculate the loss
        output_img = AutoEncoder(input_img)
        loss2 = criterion(output_img, input_img)
        
        # Perform backward pass and update model parameters
        loss2.backward()
        optimizer.step()
        
        # Store the batch loss for this epoch
        batchloss.append(loss2.item())  

    # Calculate and store the average loss for the current epoch
    epoch_loss = sum(batchloss) / len(batchloss)
    trainlosses.append(epoch_loss)
    
    # Print the loss every 200 epochs for progress monitoring
    if epoch % 200 == 0:
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

# Calculate and print the total training time
end_time = time.time()
print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# Plotting Loss vs. Epoch
plt.plot(trainlosses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

############################## Testing Model ###############################

# Set the Autoencoder model to evaluation mode
AutoEncoder.eval()

# Define a transformation to convert tensor images to PIL format
to_pil = transforms.ToPILImage()

# Loop over each image in the test dataset
for n, img in enumerate(pixeldata_test):
    # Select a random index for the image batch and move it to the device
    random_index = random.randint(0, len(img) - 1)
    img = img[random_index].to(device)

    # Perform forward pass without gradient calculation
    with torch.no_grad():
        output = AutoEncoder(img.unsqueeze(0)).to("cpu") 
        
        # Set up a subplot for original and reconstructed images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Display the original test image
        axs[0].imshow(img.cpu().squeeze(), cmap="coolwarm")
        axs[0].set_title('Test image')

        # Display the reconstructed image from the decoder
        axs[1].imshow(output.detach().cpu().squeeze(), cmap="coolwarm")
        axs[1].set_title('Decoder Image')

        # Show the plot
        plt.show()

