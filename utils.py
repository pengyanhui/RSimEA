import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as func
from torch.utils.data import DataLoader

from dataloader import TestDataset
from config import config


def set_logger():
    log_file = os.path.join(config.save_path, "train.log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_file,
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_r_hts(triples, un_rels):
    triples_1, triples_2 = triples
    un_rels_1, un_rels_2 = un_rels
    r_ht_1 = {}
    r_ht_2 = {}
    for triple in triples_1:
        h, r, t = triple
        if r not in un_rels_1:
            continue
        if r not in r_ht_1:
            r_ht_1[r] = set()
        r_ht_1[r].add((h, t))
    for triple in triples_2:
        h, r, t = triple
        if r not in un_rels_2:
            continue
        if r not in r_ht_2:
            r_ht_2[r] = set()
        r_ht_2[r].add((h, t))
    ht_1 = []
    ht_2 = []
    for r in un_rels_1:
        ht_1.append(r_ht_1[r])
    for r in un_rels_2:
        ht_2.append(r_ht_2[r])

    return ht_1, ht_2


def save_model(model, optimizer, save_vars):
    # 保存 config
    config_dict = vars(config)
    with open(os.path.join(config.save_path, "config.json"), 'w') as fjson:
        json.dump(config_dict, fjson)
    # 保存某些变量、模型参数、优化器参数
    torch.save(
        {
            **save_vars,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        },
        os.path.join(config.save_path, "checkpoint")
    )
    # 保存 embedding
    ent_embd = model.ent_embd.weight.detach().cpu().numpy()
    np.save(
        os.path.join(config.save_path, "ent_embd"),
        ent_embd
    )
    rel_embd = model.rel_embd.weight.detach().cpu().numpy()
    np.save(
        os.path.join(config.save_path, "rel_embd"),
        rel_embd
    )


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info("%s %s at step %d: %f" % (mode, metric, step, metrics[metric]))


def common(pairs, s1, s2):
    # s1 和 s2 的匹配 (h, t) 数量
    cnt = 0.0
    for r, item in enumerate(s1):
        h, t = item
        if h in pairs:
            h = pairs[h]
        if t in pairs:
            t = pairs[t]
        if (h, t) in s2:
            cnt += 1.0
    return cnt


def relation_seeds(pairs, r_ht_1, r_ht_2, un_rels):
    un_rels_1, un_rels_2 = un_rels
    similarity = np.zeros((len(r_ht_1), len(r_ht_2)), dtype=np.float32)
    for i, hts_1 in enumerate(r_ht_1):
        for j, hts_2 in enumerate(r_ht_2):
            similarity[i][j] = 2 * common(pairs, hts_1, hts_2) / (len(hts_1) + len(hts_2))

    similarity = torch.from_numpy(similarity)
    if config.cuda:
        similarity = similarity.cuda()
    max_v_1, max_idx_1 = torch.max(similarity, dim=1)
    max_v_2, max_idx_2 = torch.max(similarity, dim=0)
    rel_seeds = {}
    for i, r_1 in enumerate(un_rels_1):
        if max_v_1[i].item() > config.theta:
            idx_r_2 = max_idx_1[i]
            if max_idx_2[idx_r_2] == i:
                rel_seeds[r_1] = un_rels_2[idx_r_2]
                rel_seeds[un_rels_2[idx_r_2]] = r_1
    return rel_seeds


def entity_seeds(model, un_ents):
    model.eval()

    un_ents_1, un_ents_2 = un_ents
    un_ents_1 = torch.tensor(un_ents_1)
    un_ents_2 = torch.tensor(un_ents_2)
    if config.cuda:
        un_ents_1 = un_ents_1.cuda()
        un_ents_2 = un_ents_2.cuda()

    ents_1_embd = model.ent_embd(un_ents_1)
    ents_2_embd = model.ent_embd(un_ents_2)

    ents_loader = DataLoader(
        TestDataset(un_ents[0]),
        batch_size=config.test_batch_size,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TestDataset.collate_fn
    )

    seeds = {}
    align_e_1 = []
    align_e_2 = []
    step = 0
    total_step = un_ents_1.size(0) // config.test_batch_size
    with torch.no_grad():
        for ents in ents_loader:
            if config.cuda:
                ents = ents.cuda()

            batch_embd = model.ent_embd(ents).unsqueeze(dim=1)
            distances = torch.norm(batch_embd - ents_2_embd, p=1, dim=-1)
            min_v, min_idx = torch.min(distances, dim=-1)

            batch_size = ents.size(0)
            for i in range(batch_size):
                if min_v[i].item() < config.delta:
                    idx_ent_2 = min_idx[i]
                    ent_2 = un_ents_2[idx_ent_2]
                    embd_2 = model.ent_embd(ent_2)
                    d = torch.norm(embd_2 - ents_1_embd, p=1, dim=-1)
                    ent_1 = un_ents_1[torch.argmin(d).item()]
                    if ent_1.item() == ents[i].item():
                        seeds[ents[i].item()] = un_ents_2[min_idx[i]].item()
                        seeds[un_ents_2[min_idx[i]].item()] = ents[i].item()
                        align_e_1.append(ents[i].item())
                        align_e_2.append(un_ents_2[min_idx[i]].item())

            if step % config.test_log_step == 0:
                logging.info("Generating seeds... (%d/%d)" % (step, total_step))
            step += 1
        logging.info("Find %d pairs seeds" % (len(seeds) // 2))

    return seeds, align_e_1, align_e_2


def new_triples(triples, rel_seeds, ent_seeds):
    triples_1, triples_2 = triples
    new_triples_1 = []
    new_triples_2 = []
    for triple in triples_1:
        h, r, t = triple
        if r in rel_seeds:
            new_triples_1.append((h, rel_seeds[r], t))
        if h in ent_seeds:
            new_triples_1.append((ent_seeds[h], r, t))
        if t in ent_seeds:
            new_triples_1.append((h, r, ent_seeds[t]))
    for triple in triples_2:
        h, r, t = triple
        if r in rel_seeds:
            new_triples_2.append((h, rel_seeds[r], t))
        if h in ent_seeds:
            new_triples_2.append((ent_seeds[h], r, t))
        if t in ent_seeds:
            new_triples_2.append((h, r, ent_seeds[t]))
    logging.info("new_triples_1 num : %d" % len(new_triples_1))
    logging.info("new_triples_2 num : %d" % len(new_triples_2))
    return triples_1 + new_triples_1, triples_2 + new_triples_2


def get_optim(model, lr):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    return optimizer


def train_step(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    pos_sample, neg_sample, weight, mode = data
    if config.cuda:
        pos_sample = pos_sample.cuda()
        neg_sample = neg_sample.cuda()
        weight = weight.cuda()

    pos_score = model(pos_sample)
    neg_score = model(pos_sample, neg_sample, mode)
    pos_score = func.softplus(pos_score, beta=config.beta).squeeze(dim=1)
    neg_score = (func.softmax(- neg_score * config.alpha, dim=1).detach()
                 * func.softplus(-neg_score, beta=config.beta)).sum(dim=1)

    pos_sample_loss = (weight * pos_score).sum() / weight.sum()
    neg_sample_loss = (weight * neg_score).sum() / weight.sum()
    loss = (pos_sample_loss + neg_sample_loss) / 2

    regularization_log = {}
    ent_reg = torch.norm(model.ent_embd.weight, p=2, dim=-1).mean()
    loss += ent_reg * config.regularization
    regularization_log["ent_reg"] = ent_reg.item()

    loss.backward()
    optimizer.step()

    log = {
        **regularization_log,
        "pos_sample_loss": pos_sample_loss.item(),
        "neg_sample_loss": neg_sample_loss.item(),
        "loss": loss.item()
    }
    return log


def test_step(model, test_pairs, un_ents):
    model.eval()

    test_dataloader = DataLoader(
        TestDataset(test_pairs),
        batch_size=config.test_batch_size,
        num_workers=max(0, config.cpu_num // 3),
        collate_fn=TestDataset.collate_fn
    )

    un_ents_1 = model.ent_embd(torch.tensor(un_ents[0]).cuda())
    un_ents_2 = model.ent_embd(torch.tensor(un_ents[1]).cuda())

    logs = []
    step = 0
    total_step = len(test_pairs) // config.test_batch_size
    with torch.no_grad():
        for alignment in test_dataloader:
            if config.cuda:
                alignment = alignment.cuda()

            batch_size = alignment.size(0)

            ents_1 = alignment[:, 0]
            ents_2 = alignment[:, 1]
            ents_1_embd = model.ent_embd(ents_1)
            ents_2_embd = model.ent_embd(ents_2)
            true_score = torch.norm(ents_1_embd - ents_2_embd, p=1, dim=-1).unsqueeze(dim=-1)

            score1 = torch.norm(ents_1_embd.unsqueeze(dim=1) - un_ents_2, p=1, dim=-1)
            score2 = torch.norm(ents_2_embd.unsqueeze(dim=1) - un_ents_1, p=1, dim=-1)

            ranks_1 = torch.sum(torch.lt(score1, true_score), dim=-1)
            ranks_2 = torch.sum(torch.lt(score2, true_score), dim=-1)

            for i in range(batch_size):
                # Notice that argsort is not ranking
                # ranking + 1 is the true ranking used in evaluation metrics
                ranking_1 = 1 + ranks_1[i].item()
                ranking_2 = 1 + ranks_2[i].item()
                result = {
                    "MRR": 1.0 / ranking_1,
                    "MR": float(ranking_1),
                    "HITS@1": 1.0 if ranking_1 <= 1 else 0.0,
                    "HITS@3": 1.0 if ranking_1 <= 3 else 0.0,
                    "HITS@10": 1.0 if ranking_1 <= 10 else 0.0,
                }
                logs.append(result)
                result = {
                    "MRR": 1.0 / ranking_2,
                    "MR": float(ranking_2),
                    "HITS@1": 1.0 if ranking_2 <= 1 else 0.0,
                    "HITS@3": 1.0 if ranking_2 <= 3 else 0.0,
                    "HITS@10": 1.0 if ranking_2 <= 10 else 0.0,
                }
                logs.append(result)

            if step % config.test_log_step == 0:
                logging.info("Evaluating the model... (%d/%d)" % (step, total_step))
            step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
    return metrics
