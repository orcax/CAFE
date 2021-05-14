from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import logging
import logging.handlers
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

# from datasets import AmazonDataset
from my_knowledge_graph import *
from data_utils import OnlinePathLoader, OnlinePathLoaderWithMPSplit, KGMask
from symbolic_model import EntityEmbeddingModel, SymbolicNetwork, create_symbolic_model
from utils import *
# from const import *

logger = None


def set_logger(logname):
    global logger
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def train(args):
    dataloader = OnlinePathLoader(args.dataset, args.batch_size, topk=args.topk_candidates)
    metapaths = dataloader.kg.metapaths

    model = create_symbolic_model(args, dataloader.kg, train=True)
    params = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f'Trainable parameters: {params}')
    logger.info('==================================')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_steps = args.epochs * dataloader.total_steps
    steps = 0
    smooth_loss = []
    smooth_reg_loss = []
    smooth_rank_loss = []
    train_writer = SummaryWriter(args.log_dir)

    model.train()
    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Update learning rate
            lr = args.lr * max(1e-4, 1.0 - steps / total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # pos_paths: [bs, path_len], neg_paths: [bs, n, path_len]
            mpid, pos_paths, neg_pids = dataloader.get_batch()
            pos_paths = torch.from_numpy(pos_paths).to(args.device)
            neg_pids = torch.from_numpy(neg_pids).to(args.device)

            optimizer.zero_grad()
            reg_loss, rank_loss = model(metapaths[mpid], pos_paths, neg_pids)
            train_loss = reg_loss + args.rank_weight * rank_loss
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            smooth_loss.append(train_loss.item())
            smooth_reg_loss.append(reg_loss.item())
            smooth_rank_loss.append(rank_loss.item())

            if steps % args.steps_per_checkpoint == 0:
                smooth_loss = np.mean(smooth_loss)
                smooth_reg_loss = np.mean(smooth_reg_loss)
                smooth_rank_loss = np.mean(smooth_rank_loss)
                train_writer.add_scalar('train/smooth_loss', smooth_loss, steps)
                train_writer.add_scalar('train/smooth_reg_loss', smooth_reg_loss, steps)
                train_writer.add_scalar('train/smooth_rank_loss', smooth_rank_loss, steps)
                logger.info('Epoch/Step: {:02d}/{:08d} | '.format(epoch, steps) +
                            'LR: {:.5f} | '.format(lr) +
                            'Smooth Loss: {:.5f} | '.format(smooth_loss) +
                            'Reg Loss: {:.5f} | '.format(smooth_reg_loss) +
                            'Rank Loss: {:.5f} | '.format(smooth_rank_loss))
                smooth_loss = []
                smooth_reg_loss = []
                smooth_rank_loss = []
            steps += 1

        torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, epoch))


def train_with_mpslit(args):
    dataloader = OnlinePathLoaderWithMPSplit(args.dataset, args.batch_size, topk=args.topk_candidates,
                                             mpsplit_ratio=args.mpsplit_ratio)
    metapaths = dataloader.kg.metapaths

    model = create_symbolic_model(args, train=True)
    logger.info('== Model parameters (Trainable) ==')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)
    logger.info('==================================')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    total_steps = args.epochs * dataloader.total_steps
    steps = 0
    smooth_loss = []
    smooth_reg_loss = []
    smooth_rank_loss = []
    train_writer = SummaryWriter(args.log_dir)

    model.train()
    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            # Update learning rate
            lr = args.lr * max(1e-4, 1.0 - steps / total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # pos_paths: [bs, path_len], neg_paths: [bs, n, path_len]
            mpid, pos_paths, neg_pids = dataloader.get_batch()
            pos_paths = torch.from_numpy(pos_paths).to(args.device)
            neg_pids = torch.from_numpy(neg_pids).to(args.device)

            optimizer.zero_grad()
            reg_loss, rank_loss = model(metapaths[mpid], pos_paths, neg_pids)
            train_loss = reg_loss + args.rank_weight * rank_loss
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            smooth_loss.append(train_loss.item())
            smooth_reg_loss.append(reg_loss.item())
            smooth_rank_loss.append(rank_loss.item())

            if steps % args.steps_per_checkpoint == 0:
                smooth_loss = np.mean(smooth_loss)
                smooth_reg_loss = np.mean(smooth_reg_loss)
                smooth_rank_loss = np.mean(smooth_rank_loss)
                train_writer.add_scalar('train/smooth_loss', smooth_loss, steps)
                train_writer.add_scalar('train/smooth_reg_loss', smooth_reg_loss, steps)
                train_writer.add_scalar('train/smooth_rank_loss', smooth_rank_loss, steps)
                logger.info('Epoch/Step: {:02d}/{:08d} | '.format(epoch, steps) +
                            'LR: {:.5f} | '.format(lr) +
                            'Smooth Loss: {:.5f} | '.format(smooth_loss) +
                            'Reg Loss: {:.5f} | '.format(smooth_reg_loss) +
                            'Rank Loss: {:.5f} | '.format(smooth_rank_loss))
                smooth_loss = []
                smooth_reg_loss = []
                smooth_rank_loss = []
            steps += 1

        torch.save(model.state_dict(), '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, epoch))


def infer_paths(args):
    model = create_symbolic_model(args, train=False)

    train_labels = load_labels(args.dataset, 'train')
    train_uids = list(train_labels.keys())
    kg = load_kg(args.dataset)
    kg_mask = KGMask(kg)

    predicts = {}
    pbar = tqdm(total=len(train_uids))
    for uid in train_uids:
        predicts[uid] = {}
        for mpid in range(len(kg.metapaths)):
            metapath = kg.metapaths[mpid]
            paths = model.infer_with_path(metapath, uid, kg_mask,
                                          excluded_pids=train_labels[uid],
                                          topk_paths=20)
            predicts[uid][mpid] = paths
        pbar.update(1)
    pickle.dump(predicts, open(args.log_dir + '/infer_path_data.pkl', 'wb'))


def main():
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beauty', help='dataset name.')
    parser.add_argument('--name', type=str, default='neural_symbolic_model', help='model name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
    parser.add_argument('--steps_per_checkpoint', type=int, default=100, help='Number of steps for checkpoint.')
    parser.add_argument('--gpu', type=int, default=1, help='gpu device.')
    parser.add_argument('--embed_size', type=int, default=100, help='knowledge embedding size.')
    parser.add_argument('--deep_module', type=boolean, default=True, help='Use deep module or not')
    parser.add_argument('--use_dropout', type=boolean, default=True, help='use dropout or not.')
    parser.add_argument('--rank_weight', type=float, default=10.0, help='weighting factor for ranking loss.')
    parser.add_argument('--topk_candidates', type=int, default=20, help='weighting factor for ranking loss.')
    # parser.add_argument('--mpsplit_ratio', type=float, default=1.0, help='user metapaths split ratio.')
    parser.add_argument('--is_train', type=boolean, default=True, help='True to train model.')
    parser.add_argument('--is_infer', type=boolean, default=False, help='True to infer paths.')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    torch.backends.cudnn.enabled = False
    set_random_seed(args.seed)

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name

    if args.is_train:
        if not os.path.isdir(args.log_dir):
            os.makedirs(args.log_dir)
        set_logger(args.log_dir + '/train_log.txt')
        logger.info(args)
        # if args.mpsplit_ratio >= 1:
        train(args)
        # else:
        #     train_with_mpslit(args)

    if args.is_infer:
        args.symbolic_model = '{}/symbolic_model_epoch{}.ckpt'.format(args.log_dir, args.epochs)
        infer_paths(args)


if __name__ == '__main__':
    main()
