import torch.nn as nn
import torch
import torch.nn.functional as F
from experiment.main import args

class Logistic(nn.Module):

    def __init__(self):
        super(Logistic,self).__init__()

        self.fc1 = nn.Linear(args.split,512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x,vat = False):

        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        if vat:
            return x
        else:
            return F.log_softmax(x,dim = 1)

Net = {i: Logistic().to(args.device) for i in range(1,args.modalities+1)}
Optimizer = {i: torch.optim.Adam(Net[i].parameters(), lr = 1e-5) for i in range(1,args.modalities+1)}
