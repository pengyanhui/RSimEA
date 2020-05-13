import logging
import os

import torch

from config import config
from dataprocessing import read_elements, read_pairs, read_triples, train_data_iterator
from kgemodel import KGEModel
from utils import set_logger, log_metrics, save_model, get_optim, train_step, test_step, get_r_hts, \
    relation_seeds, entity_seeds, new_triples


def train(model, triples, entities, un_ents, un_rels, test_pairs):
    logging.info("---------------Start Training---------------")

    ht_1, ht_2 = get_r_hts(triples, un_rels)
    rel_seeds = relation_seeds({}, ht_1, ht_2, un_rels)

    current_lr = config.learning_rate
    optimizer = get_optim(model, current_lr)
    if config.init_checkpoint:
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(os.path.join(config.save_path, "checkpoint"))
        init_step = checkpoint["step"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        if config.use_old_optimizer:
            current_lr = checkpoint["current_lr"]
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        init_step = 1

    training_logs = []
    train_iterator = train_data_iterator(entities, new_triples(triples, rel_seeds, {}))
    # Training Loop
    for step in range(init_step, config.max_step):
        log = train_step(model, optimizer, next(train_iterator))
        training_logs.append(log)

        # log
        if step % config.log_step == 0:
            metrics = {}
            for metric in training_logs[0].keys():
                metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
            log_metrics("Training average", step, metrics)
            training_logs.clear()

        # warm up
        if step % config.warm_up_step == 0:
            current_lr *= 0.1
            logging.info("Change learning_rate to %f at step %d" % (current_lr, step))
            optimizer = get_optim(model, current_lr)

        if step % config.update_step == 0:
            logging.info("Align entities and relations, swap parameters")
            seeds, align_e_1, align_e_2 = entity_seeds(model, un_ents)
            rel_seeds = relation_seeds(seeds, ht_1, ht_2, un_rels)
            new_entities = (entities[0] + align_e_2, entities[1] + align_e_1)
            train_iterator = train_data_iterator(new_entities, new_triples(triples, rel_seeds, seeds))
            save_variable_list = {
                "step": step,
                "current_lr": current_lr,
            }
            save_model(model, optimizer, save_variable_list)

    logging.info("---------------Test on test dataset---------------")
    metrics = test_step(model, test_pairs, un_ents)
    log_metrics("Test", config.max_step, metrics)

    logging.info("---------------Taining End---------------")


def run():
    # load entities and relations (id)
    ents_1 = read_elements(os.path.join(config.data_path, "ent_ids_1"))
    ents_2 = read_elements(os.path.join(config.data_path, "ent_ids_2"))
    rels_1 = read_elements(os.path.join(config.data_path, "rel_ids_1"))
    rels_2 = read_elements(os.path.join(config.data_path, "rel_ids_2"))

    # triples (KG_1 and KG_2)
    triples_1 = read_triples(os.path.join(config.data_path, "triples_1"))
    triples_2 = read_triples(os.path.join(config.data_path, "triples_2"))

    # seed (test) entity alignments
    test_pairs = read_pairs(os.path.join(config.data_path, "ref_ent_ids"))

    # unaligned entities (Filter out aligned entities)
    un_ents_1 = read_elements(os.path.join(config.data_path, "unaligned_ents_1"))
    un_ents_2 = read_elements(os.path.join(config.data_path, "unaligned_ents_2"))
    un_rels_1 = read_elements(os.path.join(config.data_path, "unaligned_rels_1"))
    un_rels_2 = read_elements(os.path.join(config.data_path, "unaligned_rels_2"))

    logging.info("---------------Infomation of KG_1---------------")
    logging.info("# number of triples: %d" % len(triples_1))
    logging.info("# number of entities: %d" % len(ents_1))
    logging.info("# number of relations: %d" % len(rels_1))
    logging.info("# number of unaligned entities: %d" % len(un_ents_1))
    logging.info("# number of unaligned relations: %d" % len(un_rels_1))

    logging.info("---------------Infomation of KG_2---------------")
    logging.info("# number of triples: %d" % len(triples_2))
    logging.info("# number of entities: %d" % len(ents_2))
    logging.info("# number of relations: %d" % len(rels_2))
    logging.info("# number of unaligned entities: %d" % len(un_ents_2))
    logging.info("# number of unaligned relations: %d" % len(un_rels_2))

    logging.info("----------Infomation of Entity Alignment----------")
    logging.info("#number of seed alignments: %d" % len(set(ents_1).intersection(ents_2)))
    logging.info("#number of test alignments: %d" % len(test_pairs))

    # 创建模型
    kgemodel = KGEModel(
        ent_num=max(ents_1 + ents_2) + 1,
        rel_num=max(rels_1 + rels_2) + 1
    )
    if config.cuda:
        kgemodel = kgemodel.cuda()

    logging.info("----------Model Parameter Configuration----------")
    for name, param in kgemodel.named_parameters():
        logging.info("Parameter %s: %s, require_grad = %s" % (name, str(param.size()), str(param.requires_grad)))

    # 训练
    train(
        model=kgemodel,
        triples=(triples_1, triples_2),
        entities=(ents_1, ents_2),
        un_ents=(un_ents_1, un_ents_2),
        un_rels=(un_rels_1, un_rels_2),
        test_pairs=test_pairs
    )


if __name__ == "__main__":
    set_logger()
    run()
