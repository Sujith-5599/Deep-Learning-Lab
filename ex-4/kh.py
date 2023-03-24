import torch
import torch.nn as nn  
class KSOM(nn.Module):
  def __init__(self,T,m,n,weights,alpha):
    self.T=T
    self.m=m 
    self.n=n
    self.weights=weights
    self.alpha=alpha
  def winning_neuron(self, weights, x):
    D1 = 0
    D2 = 0
    for i in range(len(x)):
      D1 = D1 + torch.pow((x[i] - weights[0][i]), 2)
      D2 = D2 + torch.pow((x[i] - weights[1][i]), 2)
      if D1 < D2:
        return 0
      else:
        return 1
  def update(self, weights, x, win, alpha):
    for i in range(len(weights)):
      weights[win][i] = weights[win][i] + alpha * (x[i] - weights[win][i])
    return weights
T = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]])
m, n = len(T), len(T[0])
weights = torch.tensor([[0.2, 0.4, 0.6, 0.8], [0.9, 0.7, 0.5, 0.3]])
alpha = 0.5
k = KSOM(T,m,n,weights,alpha)
for i in range(1):
	for j in range(m):
		x = T[j]
		win = k.winning_neuron(weights, x)
		print(k.update(weights, x, win, alpha))