import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
batch_size_train = 10
class RBF(nn.Module):
    def __init__(self):
        super().__init__()
        input_size=184500
        output_size=10
        self.variances = 0.01
        self.output_layer = nn.Linear(input_size, output_size)
    def forward(self):
        for data, target in train_loader:
          data = data.view(data.shape[0], -1)
          centers=sum(self.data)/len(self.data)
          dists = torch.cdist(data, self.centers)
          rbf_outputs = torch.exp(-0.5 * (dists ** 2) / self.variances.view(1, -1))
          print(rbf_outputs)
          print(self.output_layer(rbf_outputs))
          return self.output_layer(rbf_outputs)
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.StanfordCars('/files/',split='train',download=True,
                                    transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Resize([205,300]),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

import torch
import torchvision
n_epochs = 3
batch_size_test = 3000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.StanfordCars('/files/',split='test',download=True,
                                    transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Resize([205,300]),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

## Specify loss and optimization functions

# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# number of epochs to train the model
n_epochs = 10  # suggest training between 20-50 epochs

#model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        #optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        model=RBF()
        model.train()
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        #optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))