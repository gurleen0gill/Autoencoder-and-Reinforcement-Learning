# Advanced Machine Learning Experiments

This notebook contains solutions to a set of tasks focused on Convolutional Neural Networks (CNN), Autoencoders, and Reinforcement Learning (RL) using the MiniPong environment. The work explores object position prediction and decision-making using deep learning techniques.

## 📌 Contents

### 🔹 Question 1: Convolutional Neural Networks (CNN)
- Predicting object positions (x / x, y, z) from 15×15 MiniPong images.
- Built CNNs with multiple convolutional layers, batch normalization, and dropout.
- Achieved ~100% accuracy for both training and testing on the dataset.

### 🔹 Question 2: Autoencoder
- Created an autoencoder for unsupervised image reconstruction.
- Learned compressed representations using convolutional and transposed convolutional layers.
- Trained with MSE loss to accurately mirror input images.

### 🔹 Question 3 & 4: Reinforcement Learning (DQN)
- Developed Deep Q-Network agents for Level 2 and Level 3 MiniPong.
- Level 3 model utilized full 3D state (x, y, z) input to learn reward-maximizing actions.
- Achieved average reward > 300 in Level 3 with proper tuning.

## 🛠 Technologies Used
- Python, PyTorch
- NumPy, Matplotlib
- Custom MiniPong environment
