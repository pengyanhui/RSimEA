from torch.utils.data import DataLoader

from config import config
from dataloader import TrainDataset, BidirectionalOneShotIterator


def read_elements(file_path):
    elements = []
    with open(file_path, 'r', encoding="UTF-8") as f:
        for line in f:
            element_id = line.strip()
            elements.append(int(element_id))
    return elements


def read_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding="UTF-8") as f:
        for line in f:
            first, second = line.strip().split('\t')
            first = int(first)
            second = int(second)
            pairs.append((first, second))
    return pairs


def read_triples(file_path):
    triples = []
    with open(file_path) as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            h = int(h)
            r = int(r)
            t = int(t)
            triples.append((h, r, t))
    return triples


def train_data_iterator(entities, triples):
    entities_1, entities_2 = entities
    triples_1, triples_2 = triples
    loader_head_1 = DataLoader(
        TrainDataset(triples_1, entities_1, config.neg_size, "head-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    loader_tail_1 = DataLoader(
        TrainDataset(triples_1, entities_1, config.neg_size, "tail-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    loader_head_2 = DataLoader(
        TrainDataset(triples_2, entities_2, config.neg_size, "head-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    loader_tail_2 = DataLoader(
        TrainDataset(triples_2, entities_2, config.neg_size, "tail-batch"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TrainDataset.collate_fn
    )
    return BidirectionalOneShotIterator(loader_head_1, loader_tail_1, loader_head_2, loader_tail_2)
