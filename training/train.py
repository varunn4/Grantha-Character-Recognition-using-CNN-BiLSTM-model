#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.functional import log_softmax
from tqdm import tqdm

# Dataset path
base_dir = r"D:\PRIYANKA M\2021-2025\PH.D\RESEARCH SCHOLAR\STUD_PROJ\2025\VARUNN  (MINI PROJECT)\Dataset\Final_dataset\Final_dataset\customized_train"
categories = ['Consonants', 'Numerals', 'mansample-final', 'uyirezhuthukkal']

images, labels = [], []
label_names = []

# 1. Gather all images and assign unique integer labels
for category in categories:
    category_path = os.path.join(base_dir, category)
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder not in label_names:
                label_names.append(subfolder)
            label_index = label_names.index(subfolder)
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subfolder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 32))  # Resize images to (128, 32)
                    images.append(img)
                    labels.append([label_index])  # as sequence for CTC

# Label dictionary for output interpretation
Label_Dict = {name: i for i, name in enumerate(label_names)}
idx2char = {i: name for i, name in enumerate(label_names)}

# 2. Prepare dataset arrays
X = np.array(images).astype(np.float32) / 255.0  # Normalize pixel values
print("Shape of X before reshaping:", X.shape)  # Debug the shape

# Check the number of dimensions in X
if len(X.shape) == 3:
    X = X[:, np.newaxis, :, :]  # [N, 1, 32, 128] (add the channel dimension)
else:
    print("Unexpected shape:", X.shape)

print("Shape of X after reshaping:", X.shape)  # Debug the shape after reshaping

print("Total images collected:", len(images))
print("Categories found:", label_names)

# 3. Train/val split
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# 5. Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

# 6. CNN + BiLSTM + CTC model
class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_BiLSTM_CTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=3,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(1024, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return log_softmax(x, dim=2)

# 7. Collate for CTC
def ctc_collate(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.full(size=(len(inputs),), fill_value=16, dtype=torch.long)
    targets_flat = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(inputs), targets_flat, input_lengths, target_lengths

# 8. DataLoaders
train_loader = DataLoader(CustomDataset(X_train, Y_train), batch_size=32, shuffle=True, collate_fn=ctc_collate)
val_loader = DataLoader(CustomDataset(X_val, Y_val), batch_size=32, shuffle=False, collate_fn=ctc_collate)

# 9. Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_BiLSTM_CTC(num_classes=len(label_names)).to(device)
criterion = nn.CTCLoss(blank=len(label_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Greedy decoder
def greedy_decoder(output):
    pred = output.permute(1, 0, 2)
    pred = torch.argmax(pred, dim=2)
    decoded = []
    for p in pred:
        seq = []
        prev = -1
        for char in p:
            if char.item() != prev and char.item() != len(label_names):
                seq.append(char.item())
            prev = char.item()
        decoded.append(seq)
    return decoded

# Accuracy computation
def compute_accuracy(preds, targets):
    correct_seq = 0
    total_seq = len(preds)
    correct_chars = 0
    total_chars = 0
    for pred, true in zip(preds, targets):
        if pred == true:
            correct_seq += 1
        correct_chars += sum(p == t for p, t in zip(pred, true))
        total_chars += len(true)
    seq_acc = correct_seq / total_seq
    char_acc = correct_chars / total_chars if total_chars else 0
    return seq_acc, char_acc

# 10. Training loop
train_losses, val_losses, train_seq_accs, val_seq_accs = [], [], [], []
epochs = 150
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Training accuracy
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            all_preds.extend(decoded_preds)
            all_targets.extend(batch_targets)
    seq_acc, char_acc = compute_accuracy(all_preds, all_targets)
    train_seq_accs.append(seq_acc)

    # Validation loss & accuracy
    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets, input_lengths, target_lengths).item()
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            val_preds.extend(decoded_preds)
            val_targets.extend(batch_targets)
    val_losses.append(val_loss / len(val_loader))
    val_seq_acc, val_char_acc = compute_accuracy(val_preds, val_targets)
    val_seq_accs.append(val_seq_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Seq Acc: {seq_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}")

# 11. Save model
torch.save(model.state_dict(), "cnn_bilstm_ctc_grantha_150.pth")
print("\u2705 Model saved as cnn_bilstm_ctc_grantha_150.pth")

# 12. Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_seq_accs, label='Train Seq Acc')
plt.plot(val_seq_accs, label='Val Seq Acc')
plt.title("Sequence Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


# In[11]:


#This is for LSTM updated

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.functional import log_softmax
from tqdm import tqdm

# Dataset path
base_dir = r"D:\PRIYANKA M\2021-2025\PH.D\RESEARCH SCHOLAR\STUD_PROJ\2025\VARUNN  (MINI PROJECT)\Dataset\Final_dataset\Final_dataset\customized_train"
categories = ['Consonants', 'Numerals', 'mansample-final', 'uyirezhuthukkal']

images, labels = [], []
label_names = []

# 1. Gather all images and assign unique integer labels
for category in categories:
    category_path = os.path.join(base_dir, category)
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder not in label_names:
                label_names.append(subfolder)
            label_index = label_names.index(subfolder)
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subfolder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 32))
                    images.append(img)
                    labels.append([label_index])

Label_Dict = {name: i for i, name in enumerate(label_names)}
idx2char = {i: name for i, name in enumerate(label_names)}

X = np.array(images).astype(np.float32) / 255.0
if len(X.shape) == 3:
    X = X[:, np.newaxis, :, :]

X_train, X_val, Y_train, Y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

# 6. CNN + LSTM + CTC model (Unidirectional)
class CNN_LSTM_CTC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM_CTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=3,
                            bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return log_softmax(x, dim=2)

def ctc_collate(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.full(size=(len(inputs),), fill_value=16, dtype=torch.long)
    targets_flat = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(inputs), targets_flat, input_lengths, target_lengths

train_loader = DataLoader(CustomDataset(X_train, Y_train), batch_size=32, shuffle=True, collate_fn=ctc_collate)
val_loader = DataLoader(CustomDataset(X_val, Y_val), batch_size=32, shuffle=False, collate_fn=ctc_collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_LSTM_CTC(num_classes=len(label_names)).to(device)
criterion = nn.CTCLoss(blank=len(label_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)

def greedy_decoder(output):
    pred = output.permute(1, 0, 2)
    pred = torch.argmax(pred, dim=2)
    decoded = []
    for p in pred:
        seq = []
        prev = -1
        for char in p:
            if char.item() != prev and char.item() != len(label_names):
                seq.append(char.item())
            prev = char.item()
        decoded.append(seq)
    return decoded

def compute_accuracy(preds, targets):
    correct_seq = 0
    total_seq = len(preds)
    correct_chars = 0
    total_chars = 0
    for pred, true in zip(preds, targets):
        if pred == true:
            correct_seq += 1
        correct_chars += sum(p == t for p, t in zip(pred, true))
        total_chars += len(true)
    seq_acc = correct_seq / total_seq
    char_acc = correct_chars / total_chars if total_chars else 0
    return seq_acc, char_acc

train_losses, val_losses, train_seq_accs, val_seq_accs = [], [], [], []
epochs = 150
best_val_loss = float('inf')  # Initialize the best validation loss to infinity
patience_counter = 0          # Initialize patience counter to zero
patience = 5                 # Set patience value, e.g., 10 epochs without improvement
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"Epoch {epoch+1}/{epochs}")  # This will print the current epoch number

    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_loader))

    # Evaluation and accuracy computation...
    # Continue with the rest of your training loop

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            all_preds.extend(decoded_preds)
            all_targets.extend(batch_targets)
    seq_acc, char_acc = compute_accuracy(all_preds, all_targets)
    train_seq_accs.append(seq_acc)

    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets, input_lengths, target_lengths).item()
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            val_preds.extend(decoded_preds)
            val_targets.extend(batch_targets)
    val_losses.append(val_loss / len(val_loader))
    val_seq_acc, val_char_acc = compute_accuracy(val_preds, val_targets)
    val_seq_accs.append(val_seq_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Seq Acc: {seq_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}")
        # Early Stopping check
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_model_state_dict = model.state_dict()  # Save the best model
        patience_counter = 0  # Reset patience counter
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
        break

overall_accuracy = (train_seq_accs[-1] + val_seq_accs[-1]) / 2
print(f"\n Overall Sequence Accuracy: {overall_accuracy:.4f}")


torch.save(best_model_state_dict, "cnn_lstm_ctc_grantha_es.pth")
print("\u2705 Model saved as cnn_lstm_ctc_grantha_es.pth")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_seq_accs, label='Train Seq Acc')
plt.plot(val_seq_accs, label='Val Seq Acc')
plt.title("Sequence Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


# In[9]:


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.functional import log_softmax
from tqdm import tqdm

# Dataset path
base_dir = r"D:\PRIYANKA M\2021-2025\PH.D\RESEARCH SCHOLAR\STUD_PROJ\2025\VARUNN  (MINI PROJECT)\Dataset\Final_dataset\Final_dataset\customized_train"
categories = ['Consonants', 'Numerals', 'mansample-final', 'uyirezhuthukkal']

images, labels = [], []
label_names = []

# 1. Gather all images and assign unique integer labels
for category in categories:
    category_path = os.path.join(base_dir, category)
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder not in label_names:
                label_names.append(subfolder)
            label_index = label_names.index(subfolder)
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subfolder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 32))  # Resize images to (128, 32)
                    images.append(img)
                    labels.append([label_index])  # as sequence for CTC

# Label dictionary for output interpretation
Label_Dict = {name: i for i, name in enumerate(label_names)}
idx2char = {i: name for i, name in enumerate(label_names)}

# 2. Prepare dataset arrays
X = np.array(images).astype(np.float32) / 255.0  # Normalize pixel values
print("Shape of X before reshaping:", X.shape)  # Debug the shape

# Check the number of dimensions in X
if len(X.shape) == 3:
    X = X[:, np.newaxis, :, :]  # [N, 1, 32, 128] (add the channel dimension)
else:
    print("Unexpected shape:", X.shape)

print("Shape of X after reshaping:", X.shape)  # Debug the shape after reshaping

print("Total images collected:", len(images))
print("Categories found:", label_names)

# 3. Train/val split
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# 5. Dataset class
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

# 6. CNN + BiLSTM + CTC model
class CNN_BiLSTM_CTC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_BiLSTM_CTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(input_size=1024, hidden_size=512, num_layers=3,
                            bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(1024, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return log_softmax(x, dim=2)

# 7. Collate for CTC
def ctc_collate(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.full(size=(len(inputs),), fill_value=16, dtype=torch.long)
    targets_flat = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(inputs), targets_flat, input_lengths, target_lengths

# 8. DataLoaders
train_loader = DataLoader(CustomDataset(X_train, Y_train), batch_size=32, shuffle=True, collate_fn=ctc_collate)
val_loader = DataLoader(CustomDataset(X_val, Y_val), batch_size=32, shuffle=False, collate_fn=ctc_collate)

# 9. Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_BiLSTM_CTC(num_classes=len(label_names)).to(device)
criterion = nn.CTCLoss(blank=len(label_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Greedy decoder
def greedy_decoder(output):
    pred = output.permute(1, 0, 2)
    pred = torch.argmax(pred, dim=2)
    decoded = []
    for p in pred:
        seq = []
        prev = -1
        for char in p:
            if char.item() != prev and char.item() != len(label_names):
                seq.append(char.item())
            prev = char.item()
        decoded.append(seq)
    return decoded

# Accuracy computation
def compute_accuracy(preds, targets):
    correct_seq = 0
    total_seq = len(preds)
    correct_chars = 0
    total_chars = 0
    for pred, true in zip(preds, targets):
        if pred == true:
            correct_seq += 1
        correct_chars += sum(p == t for p, t in zip(pred, true))
        total_chars += len(true)
    seq_acc = correct_seq / total_seq
    char_acc = correct_chars / total_chars if total_chars else 0
    return seq_acc, char_acc

# 10. Training loop with Early Stopping
train_losses, val_losses, train_seq_accs, val_seq_accs, train_accs, val_accs = [], [], [], [], [], []
epochs = 150
patience = 5  # Early stopping patience
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Training accuracy
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            all_preds.extend(decoded_preds)
            all_targets.extend(batch_targets)
    seq_acc, char_acc = compute_accuracy(all_preds, all_targets)
    train_seq_accs.append(seq_acc)

    # Overall accuracy
    correct_train = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
    train_acc = correct_train / len(all_preds)
    train_accs.append(train_acc)

    # Validation loss & accuracy
    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets, input_lengths, target_lengths).item()
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            val_preds.extend(decoded_preds)
            val_targets.extend(batch_targets)
    val_losses.append(val_loss / len(val_loader))
    val_seq_acc, val_char_acc = compute_accuracy(val_preds, val_targets)
    val_seq_accs.append(val_seq_acc)

    # Overall accuracy
    correct_val = sum(1 for p, t in zip(val_preds, val_targets) if p == t)
    val_acc = correct_val / len(val_preds)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Seq Acc: {seq_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}, "
          f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")

    # Early stopping check
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        patience_counter = 0
        # Save the model with the best validation loss
        torch.save(model.state_dict(), "grantha_bi_lstm_es.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# 11. Plots
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_seq_accs, label='Train Seq Acc')
plt.plot(val_seq_accs, label='Val Seq Acc')
plt.title("Sequence Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title("Overall Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()


# In[12]:


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.functional import log_softmax
from tqdm import tqdm

# Dataset path
base_dir = r"D:\PRIYANKA M\2021-2025\PH.D\RESEARCH SCHOLAR\STUD_PROJ\2025\VARUNN  (MINI PROJECT)\Dataset\Final_dataset\Final_dataset\customized_train"
categories = ['Consonants', 'Numerals', 'mansample-final', 'uyirezhuthukkal']

images, labels = [], []
label_names = []

# Load images and labels
for category in categories:
    category_path = os.path.join(base_dir, category)
    for subfolder in os.listdir(category_path):
        subfolder_path = os.path.join(category_path, subfolder)
        if os.path.isdir(subfolder_path):
            if subfolder not in label_names:
                label_names.append(subfolder)
            label_index = label_names.index(subfolder)
            for img_file in os.listdir(subfolder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(subfolder_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 32))
                    images.append(img)
                    labels.append([label_index])

Label_Dict = {name: i for i, name in enumerate(label_names)}
idx2char = {i: name for i, name in enumerate(label_names)}

X = np.array(images).astype(np.float32) / 255.0
if len(X.shape) == 3:
    X = X[:, np.newaxis, :, :]

X_train, X_val, Y_train, Y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

class CNN_RNN_CTC(nn.Module):
    def __init__(self, num_classes):
        super(CNN_RNN_CTC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2))
        )
        self.rnn = nn.RNN(input_size=1024, hidden_size=512, num_layers=3, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        return log_softmax(x, dim=2)

def ctc_collate(batch):
    inputs, targets = zip(*batch)
    input_lengths = torch.full(size=(len(inputs),), fill_value=16, dtype=torch.long)
    targets_flat = torch.cat(targets)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    return torch.stack(inputs), targets_flat, input_lengths, target_lengths

train_loader = DataLoader(CustomDataset(X_train, Y_train), batch_size=32, shuffle=True, collate_fn=ctc_collate)
val_loader = DataLoader(CustomDataset(X_val, Y_val), batch_size=32, shuffle=False, collate_fn=ctc_collate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_RNN_CTC(num_classes=len(label_names)).to(device)
criterion = nn.CTCLoss(blank=len(label_names))
optimizer = optim.Adam(model.parameters(), lr=0.001)

def greedy_decoder(output):
    pred = output.permute(1, 0, 2)
    pred = torch.argmax(pred, dim=2)
    decoded = []
    for p in pred:
        seq = []
        prev = -1
        for char in p:
            if char.item() != prev and char.item() != len(label_names):
                seq.append(char.item())
            prev = char.item()
        decoded.append(seq)
    return decoded

def compute_accuracy(preds, targets):
    correct_seq = 0
    total_seq = len(preds)
    correct_chars = 0
    total_chars = 0
    for pred, true in zip(preds, targets):
        if pred == true:
            correct_seq += 1
        correct_chars += sum(p == t for p, t in zip(pred, true))
        total_chars += len(true)
    seq_acc = correct_seq / total_seq
    char_acc = correct_chars / total_chars if total_chars else 0
    return seq_acc, char_acc

train_losses, val_losses, train_seq_accs, val_seq_accs = [], [], [], []
epochs = 150
patience = 5
best_val_acc = 0
patience_counter = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Train Accuracy
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in train_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            all_preds.extend(decoded_preds)
            all_targets.extend(batch_targets)
    train_seq_acc, _ = compute_accuracy(all_preds, all_targets)
    train_seq_accs.append(train_seq_acc)

    # Validation
    val_loss, val_preds, val_targets = 0, [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets, input_lengths, target_lengths).item()
            decoded_preds = greedy_decoder(outputs)
            batch_targets = [t.tolist() for t in targets.split(1)]
            val_preds.extend(decoded_preds)
            val_targets.extend(batch_targets)
    val_losses.append(val_loss / len(val_loader))
    val_seq_acc, _ = compute_accuracy(val_preds, val_targets)
    val_seq_accs.append(val_seq_acc)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Seq Acc: {train_seq_acc:.4f}, Val Seq Acc: {val_seq_acc:.4f}")

    if val_seq_acc > best_val_acc:
        best_val_acc = val_seq_acc
        patience_counter = 0
        torch.save(model.state_dict(), "simple_rnn_grantha.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered at epoch {epoch+1}")
            break

overall_accuracy = (train_seq_accs[-1] + val_seq_accs[-1]) / 2
print(f"\n✅ Overall Sequence Accuracy: {overall_accuracy:.4f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_seq_accs, label='Train Seq Acc')
plt.plot(val_seq_accs, label='Val Seq Acc')
plt.title("Sequence Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




