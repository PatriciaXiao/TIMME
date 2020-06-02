import numpy as np
import torch
import math
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import os

from utils import multi_relation_load, save_node_pred, save_link_pred
from model.model import Classification, LinkPrediction, TIMME, TIMMEhierarchical, TIMMEsingle
from model.embedding import PartlyLearnableEmbedding, FixedFeature
from task import ClassificationTask, LinkPred_BatchTask, TIMMEManager

import random
import math

import time

import warnings
warnings.filterwarnings("ignore") # ignore the warnings

parser = argparse.ArgumentParser()
parser.add_argument('-e','--epochs', type=int, default=300,
                    help='Number of epochs to train. (default: 300)')
parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adagrad", "Adam"],
                    help='The optimizer to use. (default: Adam)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate. (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=1e-3,
                    help='Learning rate decay. (default: 1e-3)')
parser.add_argument('--min_lr', type=float, default=1e-5,
                    help='Minimum learning rate. (default: 1e-5)')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters). (default: 1e-5)')
parser.add_argument('--hidden', type=int, default=100,
                    help='Number of hidden units. (default: 100)')
parser.add_argument('--single_relation', type=int, default=0,
                    help='The single-relation task to be run by single-relation. (default: 0)')
parser.add_argument('-frd','--fixed_random_seed', default=False, action='store_true',
                    help='if random seed is fixed, we will use the fixed random seed (default: false)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (default: 0.1).')
parser.add_argument('-d','--data', type=str, default='PureP',
                    choices=["PureP", "P50", "P_20_50", "P_all"],
                    help='Dataset to use. (default: PureP)')
parser.add_argument('-r', '--relations', type=str, default=['retweet_list.csv', 'mention_list.csv', 'friend_list.csv', 'reply_list.csv', 'favorite_list.csv'], action='append',
                    help='Relations to use. (default: [\'retweet_list.csv\', \'mention_list.csv\', \'friend_list.csv\', \'reply_list.csv\', \'favorite_list.csv\'])')
parser.add_argument('-t', '--task', type=str, default="Classification", choices=["Classification", "LinkPrediction", "TIMME_SingleLink", "TIMME", "TIMME_hierarchical"],
                    help='The type of task to run with (default: Classification)')
parser.add_argument('--skip_mode', type=str, default="add", choices=["none", "add", "concat"],
                    help='Not using skip connection, using skip-connection by adding the layers layer output, or skip-connection by conactenate layers. (default: add; but not much difference if we disable it)')
parser.add_argument('-att','--attention_mode', type=str, default="self", choices=["none", "naive", "self"],
                    help='Which attention mode to use. None is none; naive is not the real-attention, but trained weight; "self" option is attention (default: self)')
parser.add_argument('-lrs','--lr_scheduler', type=str, default="none", choices=["Step", "ESL", "none"],
                    help='Which learning rate scheduler to use. (default: Step)')
parser.add_argument('-f', '--feature', type=str, default="one_hot",
                    choices=["tweets_average", "description", "status", "one_hot", "random"],
                    help='The feature to use. (default: one_hot)')
parser.add_argument("--regularization_classification", type=float, default=None,
                    help="The regularization weight for node classification. (default: None)")
parser.add_argument("--regularization", type=float, default=0.01,
                    help="The regularization weight for link prediction. (default: 0.01)")
parser.add_argument("--n_batches", type=int, default=10,
                    help="The number of batches for training link prediction task. (default: 10)")
parser.add_argument("--maximum_negative_rate", type=float, default=1.5,
                    help="The maximum negative sampling rate for training link prediction task")
parser.add_argument('--freeze_feature', default=False, action='store_true',
                    help='To freeze the feature as encoder input or not. (default: False)')

args = parser.parse_args()

CUDA = torch.cuda.is_available()
print("using cuda" if CUDA else "not using cuda")

data_path = os.path.join("../data/", args.data)
DATA = Path(data_path)

# setting random seed is not necessarily needed after we get all experiments done
# for now we are simply keeping this to simplify debugging process
if args.fixed_random_seed:
    rd_seed = 36
    np.random.seed(rd_seed)
    random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    if CUDA:
        torch.cuda.manual_seed(rd_seed)


task_with_links = ["LinkPrediction", "TIMME_hierarchical", "TIMME", "TIMME_SingleLink"]

split_links=args.task in task_with_links

feature_file_table = {
    "tweets_average": "tweet_features.npz",
    "description": "features.npz",
    "status": "features.npz",
    "one_hot": None,
    "random": None
}
assert args.feature in feature_file_table.keys(), "We don't know how to get feature {}".format(args.feature)
adjs, features, labels_info, trainable, mask, link_info, (label_map, all_id_list) = multi_relation_load(DATA, files=args.relations, \
    feature_data=args.feature, feature_file=feature_file_table[args.feature], freeze_feature=args.freeze_feature, split_links=split_links)

idx_train, idx_val, idx_test, labels = labels_info
if CUDA:
    adjs = [i.cuda(0) for i in adjs]
    labels = labels.cuda(0)
    idx_train = idx_train.cuda(0)
    idx_val = idx_val.cuda(0)
    idx_test = idx_test.cuda(0)
    labels_info = (idx_train, idx_val, idx_test, labels)

num_relations = len(args.relations)
num_adjs = len(adjs)
relations = [r.split("_")[0] for r in args.relations]

if trainable is None:
    # if features are fixed
    feature_dimension = features.shape[1]
    num_entities = features.shape[0]
    feature_generator = FixedFeature(features, cuda=CUDA)
else:
    feature_dimension = features.embedding_dim
    num_entities = features.weight.shape[0]
    feature_generator = PartlyLearnableEmbedding(features.num_embeddings, features, trainable, mask, cuda=CUDA)

hidden_size = args.hidden
num_classes = labels.max().item() + 1

if args.task == "Classification":
    model = Classification(num_relations,
            num_entities,
            num_adjs,
            feature_dimension,
            hidden_size,
            num_classes,
            args.dropout,
            regularization=args.regularization_classification,
            skip_mode=args.skip_mode,
            attention_mode=args.attention_mode,
            trainable_features=trainable)
elif args.task == "LinkPrediction":
    model = LinkPrediction(num_relations,
            num_entities,
            num_adjs,
            feature_dimension,
            hidden_size,
            args.dropout,
            relations,
            regularization=args.regularization,
            skip_mode=args.skip_mode,
            attention_mode=args.attention_mode,
            trainable_features=trainable)
elif args.task == "TIMME_hierarchical":
    model = TIMMEhierarchical(num_relations,
            num_entities,
            num_adjs,
            feature_dimension,
            hidden_size,
            num_classes,
            args.dropout,
            relations,
            regularization=args.regularization,
            skip_mode=args.skip_mode,
            attention_mode=args.attention_mode,
            trainable_features=trainable)
elif args.task == "TIMME":
    model = TIMME(num_relations,
            num_entities,
            num_adjs,
            feature_dimension,
            hidden_size,
            num_classes,
            args.dropout,
            relations,
            regularization=args.regularization,
            skip_mode=args.skip_mode,
            attention_mode=args.attention_mode,
            trainable_features=trainable)
elif args.task == "TIMME_SingleLink":
    model = TIMMEsingle(num_relations,
            num_entities,
            num_adjs,
            feature_dimension,
            hidden_size,
            num_classes,
            args.dropout,
            relations,
            regularization=args.regularization,
            skip_mode=args.skip_mode,
            attention_mode=args.attention_mode,
            trainable_features=trainable,
            relation_id=args.single_relation)
else:
    print("Fatal Error: Task {} not implemented yet".format(args.task))
    exit(0)
cnt = 0
for i in model.parameters():
    cnt+=1

if args.optimizer == "Adagrad":
    optimizer = optim.Adagrad(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay,
                       lr_decay=args.lr_decay)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
else:
    print("Fatal Error: Optimizer {} not recognized.".format(args.optimizer))
    exit(0)

if CUDA:
    model.gcn.cuda()
    model.cuda()


if args.task == "Classification":
    task = ClassificationTask(model, feature_generator, adjs, args.lr, args.weight_decay, lr_scheduler=args.lr_scheduler, min_lr=args.min_lr, epochs=args.epochs)
    task.load_data(*labels_info)
    task.process_data()
    task.run(args.epochs)
    all_pred = task.get_pred()
    save_node_pred(all_pred, args.data, label_map, all_id_list, task="classification")
elif args.task == "LinkPrediction": # This option is somewhat multi-task
    task = LinkPred_BatchTask(model, feature_generator, adjs, args.lr, args.weight_decay, lr_scheduler=args.lr_scheduler, min_lr=args.min_lr, epochs=args.epochs, n_batches=args.n_batches, cuda=CUDA, negative_rate = args.maximum_negative_rate, max_epochs=args.epochs)
    task.load_data(*link_info)
    task.process_data()
    task.run(args.epochs)
elif args.task in ["TIMME", "TIMME_hierarchical", "TIMME_SingleLink"]:
    task = TIMMEManager(model, feature_generator, adjs, args.lr, args.weight_decay, lr_scheduler=args.lr_scheduler, min_lr=args.min_lr, epochs=args.epochs, n_batches=args.n_batches, cuda=CUDA, negative_rate = args.maximum_negative_rate, max_epochs=args.epochs)
    task.load_data(*labels_info, *link_info)
    task.run(args.epochs)
    all_pred = task.get_pred()
    node_pred, link_pred = all_pred
    save_node_pred(node_pred, args.data, label_map, all_id_list, task=args.task)
    save_link_pred(link_pred, args.data, relations, all_id_list, task=args.task)
    if args.task in ["TIMME_hierarchical"]:
        print("Architecture 2, lambda value: {} for relations: {}".format(model.attention_weight, " ".join(relations)))





