import torch
import numpy as np
import pandas as pd
import net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_csv = pd.read_csv('./data/train.csv')

model = net.Net()
model.to(device)

# traing the model with train_csv
model.train()
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0.0
    for i in range(len(train_csv)):
        # Separate features and label
        x = train_csv.iloc[i, 1:].values.astype(np.float32)  # Features (all columns except the first)
        y = train_csv.iloc[i, 0]  # Label (first column)
        
        # Convert to tensors and reshape for CNN
        x_tensor = torch.tensor(x, device=device).view(-1, 1, 28, 28)  # Reshape x to [1, 1, 28, 28]
        y_tensor = torch.tensor(y, device=device, dtype=torch.long)  # y is already in the correct shape
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_tensor)
        loss_val = loss(outputs, y_tensor.view(-1))  # y_tensor.view(-1) to ensure it's compatible with loss function
        
        loss_val.backward()
        optimizer.step()
        
        total_loss += loss_val.item()  # Accumulate the loss

    average_loss = total_loss / len(train_csv)  # Compute the average loss
    print(f'Epoch: {epoch}, Average Loss: {average_loss}')  

# save the model
torch.save(model.state_dict(), './model.pth')