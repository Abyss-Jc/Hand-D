import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



#Model class
class GestureMLP(nn.Module):
    def __init__(self, inputFeatures, gestures):
        super(GestureMLP, self).__init__()

        self.fc1 = nn.Linear(inputFeatures, 128)
        self.relu1 = nn.ReLU()

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.output = nn.Linear(64, gestures)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.output(x)

        return x



class GestureDataset(Dataset):

    

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        self.label_map = {
            'Fist':0,
            'Index_Finger':1,
            'Ruler':2,
            'Thumb_Up':3,
            'Idle':4
        }

        feature_cols = [f"feat_{i}" for i in range(69)]
    
        self.features = self.data[feature_cols].values

        self.labels = self.data['label'].map(self.label_map).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        x_raw = self.features[idx]
        y_raw = self.labels[idx]

        x_tensor = torch.tensor(x_raw, dtype = torch.float32) 
        y_tensor = torch.tensor(y_raw, dtype = torch.long)

        return x_tensor, y_tensor




model = GestureMLP(69, 5)
print(model)

dataset = GestureDataset('datasets/gesture_dataset.csv')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)

features_batch, labels_batch = next(iter(train_loader))

print(f"Features batch shape: {features_batch.shape}")
print(f"Labels batch shape: {labels_batch.shape}")


device = torch.device("cuda" if torch.cuda.is_available() else cpu)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=0.001)

model.to(device)

EPOCHS = 30

print(f"Starting training on {device} for {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss=0.0
    correct_train=0
    total_train=0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
       
        loss.backward()
        
      
        optimizer.step()
        
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_acc = 100 * correct_train / total_train
    
    # --- 2. VALIDATION PHASE ---
    model.eval() 
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad(): 
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Track validation accuracy
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_acc = 100 * correct_val / total_val
    
    # Print statistics for this epoch
    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% || "
          f"Val Loss: {val_loss/len(test_loader):.4f} | Val Acc: {val_acc:.2f}%")

print("\nTraining Complete!")

# ==========================================
# SAVE THE MODEL
# ==========================================
MODEL_SAVE_PATH = "models/gesture_mlp.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to '{MODEL_SAVE_PATH}'")




