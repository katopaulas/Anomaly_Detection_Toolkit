import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # class weigths [a0,a1..,an]
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        targets = targets.int()
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss

class simple_nn(torch.nn.Module):
    def __init__(self, input_features, out_features,H_SIZE=64):
        super(simple_nn, self).__init__()
        
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features,H_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(H_SIZE,H_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(H_SIZE,out_features),
            )

    def forward(self,x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x
