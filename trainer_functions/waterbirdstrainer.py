import torch
import torch.nn as nn
import torch.nn.functional as F




def evaluate_waterbirds(loader, model, device):
  #if loader.dataset.train:
  #  print('Checking accuracy on validation set')
  #else:
  #  print('Checking accuracy on test set')   
  num_correct = 0
  num_samples = 0
  dtype = torch.float
  ltype = torch.long
  model.eval()  # set model to evaluation mode
  with torch.no_grad():
    for x, y in loader:

      x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
      y = y.to(device=device, dtype=ltype)
      scores = model(x)
      _, preds = scores.max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  return acc

def train_waterbirds(model, train_loader, val_loader, optimizer, scheduler, device, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - train_loader: A PyTorch DataLoader that will yield training data.
    - val_loader: A PyTorch DataLoader that will yield validation data.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    - device: torch.device("cuda"), torch.device("mps") [for mac] or torch.device("cpu")
    
    Returns: Accuracy.
    """
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    acc = 0
    dtype = torch.float
    ltype = torch.long
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        scheduler.step()


        print('Epoch %d, loss = %.4f, lr %.8f' % (e, loss.item(), scheduler.get_lr()[0]))
        acc = evaluate_waterbirds(val_loader, model, device)

        print()
    return acc 