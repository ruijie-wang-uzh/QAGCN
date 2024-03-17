import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor

class BoWClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_hops: int):
        super(BoWClassifier, self).__init__()
        self.linear1 = nn.Linear(in_features=vocab_size, out_features=vocab_size//2)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=vocab_size//2, out_features=vocab_size//2)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=vocab_size//2, out_features=num_hops)

    def forward(self, que_vecs: FloatTensor) -> FloatTensor:
        tmp = self.linear1(que_vecs)
        tmp = self.relu1(tmp)
        tmp = self.linear2(tmp)
        tmp = self.relu2(tmp)
        tmp = self.linear3(tmp)
        return F.log_softmax(input=tmp, dim=1)
