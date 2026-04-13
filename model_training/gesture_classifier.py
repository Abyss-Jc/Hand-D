import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader




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


model = GestureMLP(69, 4)

print(model)

class GestureDataset(Dataset):

    

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
        self.label_map = {
            'Fist':0,
            'Index_Finger':1,
            'Ruler_Gesture':2,
            'Thumb_Up':3
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


dataset = GestureDataset('datasets/gesture_dataset.csv')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(dataset = train_dataset, batch_size=32, shuffle=False)

features_batch, labels_batch = next(iter(train_loader))

print(f"Features batch shape: {features_batch.shape}")
print(f"Labels batch shape: {labels_batch.shape}")