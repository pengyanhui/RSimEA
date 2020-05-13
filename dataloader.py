import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, entities, neg_size, mode):
        self.triples = triples
        self.entities = np.array(entities)
        self.neg_size = neg_size
        self.mode = mode
        self.count_t, self.count_h = self.count_frequency(self.triples)
        self.true_heads, self.true_tails = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        pos_sample = self.triples[idx]
        head, relation, tail = pos_sample
        if self.mode == "head-batch":
            weight = self.count_h[(relation, tail)]
        else:
            weight = self.count_t[(head, relation)]
        weight = torch.tensor([1.0 / weight ** 0.5])

        neg_list = []
        neg_num = 0
        while neg_num < self.neg_size:  # 采 neg_size 个负样本
            neg_sample = np.random.choice(self.entities, size=self.neg_size * 2)
            if self.mode == "head-batch":
                mask = np.in1d(  # neg_sample 在 true_head 中, 则相应的位置为 True
                    neg_sample,
                    self.true_heads[(relation, tail)],  # np.array 的意义
                    assume_unique=True,
                    invert=True  # True 变 False, False 变 True
                )
            elif self.mode == "tail-batch":
                mask = np.in1d(
                    neg_sample,
                    self.true_tails[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError("Training batch mode %s not supported" % self.mode)
            neg_sample = neg_sample[mask]  # 滤除 False 处的值
            neg_list.append(neg_sample)  # 内容为 array
            neg_num += neg_sample.size

        neg_sample = np.concatenate(neg_list)[:self.neg_size]  # 合并, 去掉多余的
        neg_sample = torch.from_numpy(neg_sample).long()
        pos_sample = torch.tensor(pos_sample)

        return pos_sample, neg_sample, weight, self.mode

    @staticmethod
    def collate_fn(data):
        pos_sample = torch.stack([_[0] for _ in data], dim=0)
        neg_sample = torch.stack([_[1] for _ in data], dim=0)
        weight = torch.cat([_[2] for _ in data], dim=0)
        return pos_sample, neg_sample, weight, data[0][3]

    @staticmethod
    def count_frequency(triples, start=4):
        """
        The frequency will be used for subsampling like word2vec
        """
        count_hr = {}
        count_rt = {}
        for head, relation, tail in triples:
            if (head, relation) not in count_hr:
                count_hr[(head, relation)] = start
            else:
                count_hr[(head, relation)] += 1

            if (relation, tail) not in count_rt:
                count_rt[(relation, tail)] = start
            else:
                count_rt[(relation, tail)] += 1
        return count_hr, count_rt

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}
        # 统计 {hr:true_tails, rt:true_heads}
        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = set()
            true_tail[(head, relation)].add(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = set()
            true_head[(relation, tail)].add(head)
        # 变 np.array, 利于过滤负采样中的正样本
        for rt in true_head:
            true_head[rt] = np.array(list(true_head[rt]))
        for hr in true_tail:
            true_tail[hr] = np.array(list(true_tail[hr]))
        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, alignments):
        self.alignments = alignments

    def __len__(self):
        return len(self.alignments)

    def __getitem__(self, idx):
        return torch.tensor(self.alignments[idx])

    @staticmethod
    def collate_fn(data):
        return torch.stack([_ for _ in data], dim=0)


class BidirectionalOneShotIterator(object):
    def __init__(self, loader_head_1, loader_tail_1, loader_head_2, loader_tail_2):
        self.iterator_head_1 = self.one_shot_iterator(loader_head_1)
        self.iterator_tail_1 = self.one_shot_iterator(loader_tail_1)
        self.iterator_head_2 = self.one_shot_iterator(loader_head_2)
        self.iterator_tail_2 = self.one_shot_iterator(loader_tail_2)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 4 == 0:
            data = next(self.iterator_head_1)
        elif self.step % 4 == 1:
            data = next(self.iterator_tail_1)
        elif self.step % 4 == 2:
            data = next(self.iterator_head_2)
        else:
            data = next(self.iterator_tail_2)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        """
        顾名思义, 一次发射一个 batch
        """
        while True:
            for data in dataloader:
                yield data
