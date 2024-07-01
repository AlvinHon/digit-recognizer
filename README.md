# Digit Recognizer

This repository contains the solution to the competition [Digital Recognizer](https://www.kaggle.com/competitions/digit-recognizer/) on Kaggle.

Mainly used **Github Copilot** to build this solution without bothering the details in Machine Learning. It was built in a short time and the score is 0.92314.

Before using `Copilot`, the solution in my mind is a Nerual Network implemented in Pytorch. Here is how I prompted to `Copilot`.

**Comment: Create NN model for the dataset of digit recognition**

(Auto Code Generation):
```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(784, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```

**Ask: use cuda for this model**

[Copilot's response](./response/q1.md)

**Ask: fix the error RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)**

[Copilot's response](./response/q2.md)

**Ask: fix the error RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x783 and 784x256)**

[Copilot's response](./response/q3.md)

**Ask: train_csv has 785 columns where the first column is the label. test_csv and 784 columns without label**

[Copilot's response](./response/q4.md)

**Ask: also modify for model testing**

[Copilot's response](./response/q5.md)

**Ask: the 784 is the number of pixels of an image with size 28 x 28. modify the model to use convolution nerual network**

[Copilot's response](./response/q6.md)

**Ask: also modify for the training part**

[Copilot's response](./response/q7.md)

**Ask: modify the running model part**

[Copilot's response](./response/q8.md)

**Ask: save the prediction output into a csv file with header "ImageId,Label"**

[Copilot's response](./response/q9.md)
