import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCPC(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, nlayers, dropP = 0.2, batch_size = 1):
        super(GRUCPC, self).__init__()
        self.n_layers = nlayers
        self.hidden_dim = hid_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(in_dim, hid_dim, nlayers, dropout = dropP)
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        return hidden

    def contrastive_loss(self, hidden, positive, negative):
        score  = torch.mul(hidden, positive)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        neg_score  = torch.mul(hidden, negative)
        neg_score = torch.sum(neg_score, dim=1)
        neg_log_target = F.logsigmoid(neg_score).squeeze()

        loss = log_target - neg_log_target
        return (-1*loss.sum()/self.batch_size)

    def forward(self, x, h, pos, neg):
        out, h = self.gru(x, h)
        loss = 0
        for i in range(len(pos)):
            loss += self.contrastive_loss(h, pos[i], neg[i])
        return out, h, loss
    
    def embgen(self, x, h):
        out, h = self.gru(x, h)
        return h