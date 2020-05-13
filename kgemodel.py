import torch
import torch.nn as nn

from config import config

class KGEModel(nn.Module):
    def __init__(self, ent_num, rel_num):
        super(KGEModel, self).__init__()
        self.gamma = nn.Parameter(torch.tensor(config.gamma), requires_grad=False)
        self.ent_embd = nn.Embedding(ent_num, config.dim)
        self.rel_embd = nn.Embedding(rel_num, config.dim)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.ent_embd.weight)
        nn.init.kaiming_uniform_(self.rel_embd.weight)

    def get_pos_embd(self, pos_sample):
        head = self.ent_embd(pos_sample[:, 0]).unsqueeze(dim=1)
        relation = self.rel_embd(pos_sample[:, 1]).unsqueeze(dim=1)
        tail = self.ent_embd(pos_sample[:, 2]).unsqueeze(dim=1)
        return head, relation, tail

    def get_neg_embd(self, neg_sample):
        return self.ent_embd(neg_sample)

    def forward(self, pos_sample, neg_sample=None, mode=None):
        head, relation, tail = self.get_pos_embd(pos_sample)
        if neg_sample is not None:
            neg_embd = self.get_neg_embd(neg_sample)
            if mode == "head-batch":
                score = neg_embd + (relation - tail)
            elif mode == "tail-batch":
                score = (head + relation) - neg_embd
            else:
                raise ValueError("mode %s not supported" % mode)
        else:
            score = head + relation - tail
        score = torch.norm(score, p=1, dim=-1) - self.gamma
        return score
