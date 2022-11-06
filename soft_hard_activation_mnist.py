import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def soft(x, tau = 5):
    return torch.relu(x - tau) - torch.relu(-x - tau)

# Class wrappers for ease of use with pytorch
class SoftAct(nn.Module):
    def __init__(self, in_features, tau = None):
        super(SoftAct, self).__init__()
        self.in_features = in_features
        
        if tau == None:
            self.tau = Parameter(torch.tensor(0.0))
        else:
            self.tau = Parameter(torch.tensor(tau))
        self.tau.requiresGrad = True
    
    def forward(self, z):
        return soft(z, self.tau)

def hard(x, tau = 5):
    I = (abs(x) > tau).float()   # indicator---necessary because "if x > tau: ..." will not bool entrywise
    return x * I
        
class HardAct(nn.Module):
    def __init__(self, in_features, tau = None):
        super(HardAct, self).__init__()
        self.in_features = in_features
        
        if tau == None:
            self.tau = Parameter(torch.tensor(0.0))
        else:
            self.tau = Parameter(torch.tensor(tau))
        self.tau.requiresGrad = True
    
    def forward(self, z):
        return hard(z, self.tau)

# %% Load data (try also FashionMNIST for a more challenging dataset)
training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor()
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

# %% Define the NN: two hidden layers. The tau parameters are learned too
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 10)
        
        self.t1 = SoftAct(512)      # SoftAct or HardAct
        self.t2 = SoftAct(512)
        self.t3 = SoftAct(512)
   
    def forward(self, x):
        x = x.view(x.shape[0], -1)  # flatten
        
        x = self.t1(self.fc1(x))
        x = self.t2(self.fc2(x))
        x = self.t3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim = 1)

        return x

model = NeuralNetwork()

# %%  Set loss function and optimiser; set up training/testing the model
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = 0.001)

def train(dataloader, model, loss_fn, optimiser):
    size = len(dataloader.dataset)
    model.train()   # train mode
   
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to("cpu"), y.to("cpu")
        
        pred = model(X)
        loss = loss_fn(pred, y)
       
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
       
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    # evaluation mode
    test_loss, correct = 0, 0
   
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to("cpu"), y.to("cpu")
            
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
    return test_loss    # loss returned for ease of plotting loss later

# %% Train and test model
epochs = 100
losses = np.zeros(epochs)
for t in range(epochs):
    print(f"Epoch {t + 1}\n===============")
    train(train_dataloader, model, loss_fn, optimiser)
    losses[t] = test(test_dataloader, model, loss_fn)
print("Done.")

# %% Optional: Display a sample from the test set with label and model prediction
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

model.eval()
n = np.random.randint(0, len(test_data))
x, y = test_data[n][0], test_data[n][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted}.\nActual: {actual}.")
plt.imshow(x[0], cmap = "Greys")
plt.show()



