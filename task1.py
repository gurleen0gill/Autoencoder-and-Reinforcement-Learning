## Importing all the necessary libraries
import pandas as pd
import numpy as np
import torch
#import torchvision
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim 
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
import timeit
from tqdm import tqdm

# Loading the data
y_train = pd.read_csv("traininglabels.csv", header = None)
x_train = pd.read_csv("trainingpix.csv", header= None)
y_test = pd.read_csv("testinglabels.csv", header = None)
x_test = pd.read_csv("testingpix.csv", header= None)

############################## Preprocessing ###############################

#Adjusting the indexes of the data values
y_train.iloc[:,0:2] = y_train.iloc[:,0:2]-1
y_test.iloc[:,0:2] = y_test.iloc[:,0:2]-1

#Checking whether the system has the gpu or cpu to work on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Reshaping and creating the array of the data values
x_train_tensordata = torch.tensor(x_train.values, dtype=torch.float32).to(device).reshape(-1, 1, 15, 15)
x_test_tensordata = torch.tensor(x_test.values, dtype=torch.float32).to(device).reshape(-1, 1, 15, 15)
y_train_tensordata = torch.tensor(y_train.values, dtype=torch.float32).to(device)
y_test_tensordata = torch.tensor(y_test.values, dtype=torch.float32).to(device)

#Loading the tensor data with the help of tensor data
train_data = data.TensorDataset(x_train_tensordata, y_train_tensordata)
train_dataloader = data.DataLoader(train_data, batch_size = 32, shuffle = True)

test_data = data.TensorDataset(x_test_tensordata, y_test_tensordata)
test_dataloader = data.DataLoader(test_data, batch_size = 32, shuffle = True)

############################################## Question 1 - CNN ###################################################################

############################## Part 1 ###############################

# Defining CNN model for x-coordinate prediction
class CNNXnet(nn.Module):
    def __init__(self):
        super(CNNXnet, self).__init__()

        #First Convolutional Layer with 1 input channels, 32 output channels
        self.con1 = nn.Conv2d(1, 32, kernel_size = 3)  
        #Batch normalization for 32 channel
        self.bn1 = nn.BatchNorm2d(32)    

        #Adding pooling with 2x2 window
        self.pool = nn.MaxPool2d(2, 2)   
        
        #Second Convolutional Layer with 32 input channels, 64 output channels
        self.con2 = nn.Conv2d(32, 64, kernel_size= 3, padding=1)  
        #Batch normalization for 64 channel
        self.bn2 = nn.BatchNorm2d(64)
        
        #Third Convolutional Layer with 64 input channels, 128 output channels
        self.con3 = nn.Conv2d(64, 128, kernel_size= 3, padding=1)
        #Batch normalization for 128 channel
        self.bn3 = nn.BatchNorm2d(128)

        #Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        #Adding Fully connected layer
        self.flatten_size = nn.Linear(128 * 1 * 1, 128)  

        #Output layer for 13 classes for x-coordinate
        self.fcn1 = nn.Linear(128, 13) 

    def forward(self, x):
        #First layer 
        x = self.pool(F.relu(self.bn1(self.con1(x))))
        #Second layer 
        x = self.pool(F.relu(self.bn2(self.con2(x)))) 
        #Third layer 
        x = self.pool(F.relu(self.bn3(self.con3(x))))  

        #Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1) 
        #Dropout 
        x = self.dropout(x)        

        #Fully connected layer with ReLU
        x_classx = F.relu(self.flatten_size(x))
        #Output layer for x-coordinate  
        x_classx = self.fcn1(x_classx)           

        return x_classx
    
#Saving the model
CNNXModel = CNNXnet().to(device)

############################## Training Model ###############################

#Parameters
learning_rate = 0.0001
epochs = 100
#CrossEntropyLoss for classification
criterionx = nn.CrossEntropyLoss() 
#Opitimizing
optimizerx = optim.Adam(CNNXModel.parameters(), lr=learning_rate, weight_decay=0.00001)

#Training variables
x_modelloss = []
x_train_accry = []

#Training loop
start_time = time.time()

for epoch in tqdm(range(epochs)):
    #Setting the model to training mode
    CNNXModel.train()  
    total_loss = 0
    total_samples = 0
    correct_predictions = 0

    for image, label in train_dataloader:
        #Making the gradients zero
        optimizerx.zero_grad()  
        
        #Forward pass predicting x-coordinate only and reshaping and sending labels
        image = image.view(-1, 1, 15, 15).to(device)  
        label = label.to(device)  
        
        #x-coordinate prediction
        x = CNNXModel(image)  
        
        #Compute the loss for x-coordinate
        xloss = criterionx(x, label[:, 0].long())  
        
        # Backward pass and optimization
        xloss.backward()
        optimizerx.step()
        
        #Accumulate total loss
        total_loss += xloss.item() * label.size(0)
        total_samples += label.size(0)
        
        #Accuracy calculation
        _, predicted_x = torch.max(x, 1)  # Get the predicted x class
        correct_predictions += (predicted_x == label[:, 0].long()).sum().item()  # Compare predictions with true labels
    
    #Average training loss for this epoch
    avg_loss = total_loss / total_samples

    #Training accuracy for x-coordinate
    accry_x = (correct_predictions / total_samples) * 100
    
    #Append statistics
    x_modelloss.append(avg_loss)
    x_train_accry.append(accry_x)
    
print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Training Accuracy for x: {accry_x:.2f}%')

#End of training
end_time = time.time()
print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")


############################## Testing Model ###############################

#List to store predictions for x-coordinate
test_x_predictions = []
#Set the model to evaluation mode
CNNXModel.eval()

#Initializing variables
testx_total_acc = 0
testx_pred = 0

#No gradient calculations required during evaluation
with torch.no_grad():
    for testx_image, testx_label in test_dataloader:
        #Reshaping image
        testx_image = testx_image.view(-1, 1, 15, 15).to(device)
        testx_label = testx_label.to(device)

        #Forward pass to get predictions
        xtest = CNNXModel(testx_image)

        #Get the predicted class with the highest score
        _, testx_predicted = torch.max(xtest, 1)
            
        #Calculating number of corrected predictions
        testx_pred += (testx_predicted == testx_label[:, 0].long()).sum().item()
                       
        #Updating total number of labels processed
        testx_total_acc += testx_label.size(0)

        #Storing the predictions in CPU memory for saving
        test_x_predictions.extend(testx_predicted.cpu().numpy())  

#Calculate test accuracy
testx_accry = (testx_pred / testx_total_acc) * 100

#Set model back to training mode
CNNXModel.train()

#Save predictions to a CSV file
predictions_df = pd.DataFrame({'x_pred': test_x_predictions})
predictions_df.to_csv('x_predictions3.csv', index=False)

print("Predicted x values saved to 'x_predictions3.csv'")
print(f'Test Accuracy for x: {testx_accry:.2f}%')
    
#Plot training loss over epochs
plt.plot(x_modelloss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss for x')
plt.title('Training Loss vs. Epochs')
plt.legend()
plt.show()

############################## Part 2 ######################################

# Define the CNN model for predicting x, y, z coordinates
class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()

        #First Convolutional Layer with 1 input channels, 32 output channels
        self.con1 = nn.Conv2d(1, 32, 3)
        #Max pooling with a 2x2 kernel
        self.pool = nn.MaxPool2d(2, 2)  
        #Second Convolutional Layer with 32 input channels, 64 output channels
        self.con2 = nn.Conv2d(32, 64, 3, padding=1)
        #Third Convolutional Layer with 64 input channels, 128 output channels
        self.con3 = nn.Conv2d(64, 128, 3, padding=1)
        #Dropout layer with probability of 0.15
        self.dropout = nn.Dropout(0.15)  

        #Fully connected layer to flatten output
        self.flatten_size = nn.Linear(128 * 1 * 1, 128)

        #Fully connected output layers for each coordinate (x, y, z)
        #Output layer for x-coordinate with 13 classes
        self.fcn1 = nn.Linear(128, 13) 
        #Output layer for y-coordinate with 13 classes 
        self.fcn2 = nn.Linear(128, 13)  
        #Output layer for z-coordinate with 5 classes
        self.fcn3 = nn.Linear(128, 5)   

    # Forward pass of the model
    def forward(self, x):
        #Convolution, ReLU, and pooling after layers
        ##First layer
        x = self.pool(F.relu(self.con1(x)))
        #FiSecondrst layer  
        x = self.pool(F.relu(self.con2(x)))
        #Third layer
        x = self.pool(F.relu(self.con3(x)))  

        #Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)            

        #Appling Dropout
        x = self.dropout(x)                  

        #Fully connected layer with ReLU
        x = F.relu(self.flatten_size(x))  
        #Output for x-coordinate   
        x_class = self.fcn1(x) 
        #Output for y-coordinate              
        y_class = self.fcn2(x)
        #Output for z-coordinate               
        z_class = self.fcn3(x)               

        return x_class, y_class, z_class    

# Instantiate the CNN model
CNNModel = CNNnet()

############################## Training the model ###############################

learning_rate = 0.0001
epochs = 350
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNNModel.parameters(), lr= learning_rate,weight_decay=0.000001)

## Training loop

start_time = time.time()

modelloss = []
train_accry = []

for epoch in tqdm(range(epochs)):
    CNNModel.train()
    train_total_acc = 0
    corrected_acc = 0
    x_pred = 0
    y_pred = 0
    z_pred = 0
    per_epoch_loss = []
    for n, (image, label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        image = image.view(-1, 1, 15, 15).to(device)
        label = label.to(device)

        # calculate loss for x,y,z seperately
        image =  image.view(-1, 1, 15, 15)
        x,y,z = CNNModel(image)
    
        loss_x = criterion(x, label[:,0].long())
        loss_y = criterion(y, label[:,1].long())
        loss_z = criterion(z, label[:,2].long())

        loss = loss_x + loss_y + loss_z

        loss.backward()
        optimizer.step()

        per_epoch_loss.append(loss.item())

        # Train Accuracy calculation
        _, predicted_x = torch.max(x, 1)
        _, predicted_y = torch.max(y, 1)
        _, predicted_z = torch.max(z, 1)

        x_pred += (predicted_x == label[:, 0].long()).sum().item()
        y_pred += (predicted_y == label[:, 1].long()).sum().item()
        z_pred += (predicted_z == label[:, 2].long()).sum().item()

        corrected_acc += ((predicted_x == label[:, 0].long()) & 
                    (predicted_y == label[:, 1].long()) & 
                    (predicted_z == label[:, 2].long())).sum().item()
        train_total_acc += label.size(0)

    x_accry = (x_pred / train_total_acc) * 100
    y_accry = (y_pred / train_total_acc) * 100
    z_accry = (z_pred / train_total_acc) * 100

    train_accrcy = 100 * corrected_acc / train_total_acc
    
    # Append the epoch accuracy to the list
    train_accry.append(train_accrcy)
    
    modelloss.append(np.mean(per_epoch_loss))

  
end_time = timeit.default_timer()
print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.\n")


print(f'Accuracy for x: {x_accry:.2f}%')
print(f'Accuracy for y: {y_accry:.2f}%')
print(f'Accuracy for z: {z_accry:.2f}%')
print(f'Accuracy for all (x, y, z correct): {train_accrcy:.2f}%')
print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.mean(per_epoch_loss):.4f}, Training Accuracy: {train_accrcy:.2f}%")

############################## Testing the model ###############################

# Initialize lists to store predictions for x, y, and z coordinates
x_predictions = []
y_predictions = []
z_predictions = []

# Set the model to evaluation mode
CNNModel.eval()

# Initialize variables to accumulate accuracy and correct predictions
test_total_acc = 0
test_corrected_acc = 0
test_x_pred = 0
test_y_pred = 0
test_z_pred = 0
test_per_epoch_loss = []

# Disable gradient calculations for evaluation to save memory and computation
with torch.no_grad():
    for test_image, test_label in test_dataloader:
        # Reshape the test images and move them to the specified device
        test_image = test_image.view(-1, 1, 15, 15).to(device)
        test_label = test_label.to(device)

        # Forward pass through the CNN model to get predictions for x, y, z coordinates
        x_test, y_test, z_test = CNNModel(test_image)

        # Extract the predicted class for each coordinate by taking the max along dimension 1
        _, test_predicted_x = torch.max(x_test, 1)
        _, test_predicted_y = torch.max(y_test, 1)
        _, test_predicted_z = torch.max(z_test, 1)

        # Accumulate correct predictions for x, y, and z coordinates
        test_x_pred += (test_predicted_x == test_label[:, 0].long()).sum().item()
        test_y_pred += (test_predicted_y == test_label[:, 1].long()).sum().item()
        test_z_pred += (test_predicted_z == test_label[:, 2].long()).sum().item()
            
        # Count instances where all x, y, and z predictions are correct
        test_corrected_acc += ((test_predicted_x == test_label[:, 0].long()) & 
                               (test_predicted_y == test_label[:, 1].long()) & 
                               (test_predicted_z == test_label[:, 2].long())).sum().item()

        # Update the total number of test samples
        test_total_acc += test_label.size(0)
            
        # Transfer predictions to CPU and store them in lists for x, y, and z coordinates
        x_predictions.extend(test_predicted_x.cpu().numpy())
        y_predictions.extend(test_predicted_y.cpu().numpy())
        z_predictions.extend(test_predicted_z.cpu().numpy())

    # Calculate accuracy for each coordinate as a percentage
    test_x_accry = (test_x_pred / test_total_acc) * 100
    test_y_accry = (test_y_pred / test_total_acc) * 100
    test_z_accry = (test_z_pred / test_total_acc) * 100

    # Calculate combined accuracy where all three coordinates are correctly predicted
    test_accrcy = 100 * test_corrected_acc / test_total_acc

    # Switch model back to training mode
    CNNModel.train()

# Create a DataFrame to store x, y, z predictions and save it to CSV
testxyz_predictions_df = pd.DataFrame({
    'x_pred': x_predictions,
    'y_pred': y_predictions,
    'z_pred': z_predictions
})
testxyz_predictions_df.to_csv('xyz_predictions7.csv', index=False)
print("Predicted x, y, z values saved to 'xyz_predictions7.csv'")

# Print out the accuracy for x, y, z individually and for all three coordinates combined
print(f'Test Accuracy for x: {test_x_accry:.2f}%')
print(f'Test Accuracy for y: {test_y_accry:.2f}%')
print(f'Test Accuracy for z: {test_z_accry:.2f}%')
print(f'Test Accuracy for all (x, y, z correct): {test_accrcy:.2f}%')

# Plot Training Loss
plt.plot(modelloss, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
