import torch
import numpy as np
import pandas as pd
import net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
model = net.Net()
model.to(device)
model.load_state_dict(torch.load('./model.pth'))

# Load the test csv
test_csv = pd.read_csv('./data/test.csv')

# Ensure the model is in evaluation mode
model.eval()

predictions = []  # Initialize an empty list for predictions
for i in range(0, len(test_csv)):
    # Separate features and label
    x = test_csv.iloc[i, :].values.astype(np.float32)  # Features (all columns)
    
    # Convert to tensors and reshape for CNN
    sample = torch.tensor(test_csv.iloc[i, :].values, device=device, dtype=torch.float32).view(-1, 1, 28, 28)
    
    # Run the model with the reshaped sample
    with torch.no_grad():  # No need to track gradients for inference
        y_pred = model(sample)
        _, predicted = torch.max(y_pred, 1)  # Use dim=1 to get the max per row
        predictions.append((i+1, predicted.item()))  # Append (ImageId, Label)
    # Print the progress
    if (i+1) % 1000 == 0:
        print(f'Finished {i+1}/{len(test_csv)} samples')

# Convert to DataFrame and save as CSV
df_predictions = pd.DataFrame(predictions, columns=["ImageId", "Label"])
df_predictions.to_csv('./predictions.csv', index=False)