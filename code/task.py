import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import accuracy, f1_score, roc_auc_score, precision_recall_score, link_accuracy, link_accuracy_fixed_thresholds
from utils import generate_masked_adjs, debug_mask

import numpy as np

from sampler import NaiveSampler as Sampler
# from sampler import MultiprocessSampler as Sampler # not working well with out model in practice --- synchronization problems causing it to be slower
from lr_scheduler import StepDecay, ESLearningDecay, NoneDecay

class TaskManager(object):
    """
    TaskManager: the base class of the task managers, handling the training and evaluation pipeline of our tasks.
    """
    def __init__(self, model, features_generator, adjs, lr, weight_decay, algorithm="Adam", fastmode=False, lr_scheduler="Step", min_lr=1e-5, epochs=600, n_batches=1):
        self.model = model
        self.features_generator = features_generator
        self.adjs = adjs
        self.fastmode = fastmode
        self.lr = lr
        self.weight_decay = weight_decay
        self.set_optimizer(algorithm=algorithm)
        self.task_name = ""
        self.set_lr_scheduler(lr_scheduler, epochs, n_batches, min_lr)
    def set_lr_scheduler(self, lr_scheduler, epochs, n_batches, min_lr):
        if lr_scheduler == "none":
            self.lr_scheduler = NoneDecay(self.lr)
        elif lr_scheduler == "Step":
            self.lr_scheduler = StepDecay(self.lr, min_lr=min_lr)
        elif lr_scheduler == "ESL":
            self.lr_scheduler = ESLearningDecay(self.lr, alpha=2.0, T=epochs, b=n_batches, min_lr=min_lr)
        else:
            print("couldn't recognize the scheduler {}".format(lr_scheduler))
            exit(0)
    def load_data(self, train_data, valid_data, test_data, labels=None):
        self.train_data_raw = train_data
        self.valid_data_raw = valid_data
        self.test_data_raw = test_data
        self.labels = labels
    def process_data(self):
        self.train_data = self.train_data_raw
        self.valid_data = self.valid_data_raw
        self.test_data = self.test_data_raw
    def set_optimizer(self, trainable_params=None, optimizer=None, algorithm="Adam"):
        if optimizer is not None:
            self.optimizer = optimizer
            return
        trainable_params = trainable_params if trainable_params is not None else self.model.parameters()
        if algorithm == "Adam":
            self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            print("optimization algorithm {} not supported".format(algorithm))
            self.optimizer = None
    def report(self, note, report_dict):
        '''
        reporting, needed for validation and testing print-outs
        '''
        print(note, end=" ")
        for k,v in report_dict.items():
            print("{}: {:.4f}".format(k, v), end=" ")
        print("")
    def loss_epoch(self, model_output):
        '''
        the function of calculating loss for backprop
        called by train_epoch
        '''
        return self.model.get_loss(model_output, self.labels, self.train_data)
    def train_epoch(self):
        '''
        training one epoch
        called by train
        '''
        # set the model to training mode
        self.model.train()
        self.optimizer.zero_grad()
        features = self.features_generator()
        output = self.model(features, self.adjs)
        loss_train = self.loss_epoch(output)
        loss_train.backward()
        self.optimizer.step()
        return {"loss_train": loss_train.item(), "features": features}
    def eval_metrics(self, model_output, data, prefix=""):
        '''
        return the dictionary of the metrics of evaluation, specific for every task
        called by evaluate
        '''
        return {}
    def evaluate(self, data, features=None, **kwargs):
        '''
        evaluation script, needed for both validation and testing
        called by train and test
        '''
        # set the model to evaluation mode
        self.model.eval()
        features = features if features is not None else self.features_generator()
        output = self.model(features, self.adjs)
        return self.eval_metrics(output, data, **kwargs)
    def train(self, epochs):
        ret_val = None
        # if optimizer hasn't been supported
        if self.optimizer is None:
            print("Using the default optimizer.")
            self.set_optimizer()
        t = time.time()
        for epoch_id in range(epochs):
            # print("Begin training epoch ({}/{})".format(epoch_id+1, epochs))
            self.lr_scheduler.update(epoch_id, self.optimizer)
            ######
            var_dict = self.train_epoch()
            if not self.fastmode:
                features = var_dict.pop("features", None)
                var_dict.update(self.evaluate(self.valid_data, features))
                ret_val = var_dict.pop("return", None)
                var_dict.update({"time": time.time() - t})
                self.report('Epoch: {:04d}/{}'.format(epoch_id+1, epochs), var_dict)
                t = time.time()
        if self.fastmode or epochs==0:
            ret_val = self.evaluate(self.valid_data).pop("return", None)
        return ret_val
    def test(self, **kwargs):
        var_dict = self.evaluate(self.test_data, **kwargs)
        ret_val = var_dict.pop("return", None)
        self.report("{} test results:".format(self.task_name), var_dict)
        return ret_val

    def run(self, epochs):
        t_start = time.time()
        print("Start Training {}".format(self.task_name))
        params = self.train(epochs)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_start))
        self.test() if params is None else self.test(**params)

class ClassificationTask(TaskManager):
    def __init__(self, model, features_generator, adjs, lr, weight_decay, algorithm="Adam", fastmode=False, lr_scheduler="Step", min_lr=1e-5, epochs=600):
        super().__init__(model, features_generator, adjs, lr, weight_decay, algorithm, fastmode, lr_scheduler, min_lr, epochs)
        self.task_name = "Node classification "
    def eval_metrics(self, model_output, data):
        acc_val = accuracy(model_output[data].cpu(), self.labels[data].cpu())
        f1_val = f1_score(model_output[data].cpu(), self.labels[data].cpu())
        return {"accuracy": acc_val.item(), "f1_score": f1_val.item()}
    def get_pred(self):
        self.model.eval()
        features = self.features_generator()
        output = self.model(features, self.adjs).cpu().detach().numpy()
        return output # as N * 2 numpy
    

class LinkPredictionTask(TaskManager):
    def __init__(self, model, features_generator, adjs, lr, weight_decay, algorithm="Adam", fastmode=False, lr_scheduler="Step", min_lr=1e-5, epochs=600, n_batches=10, n_val_batches=1, n_test_batches=1, negative_rate = 1.5, cuda=False, report_interval=0, max_epochs=100):
        super().__init__(model, features_generator, adjs, lr, weight_decay, algorithm, fastmode, lr_scheduler, min_lr, epochs, n_batches)
        self.task_name = "Link prediction "
        self.n_entities = adjs[0].size()[0]
        self.n_batches=n_batches
        self.n_val_batches = n_val_batches
        self.n_test_batches = n_test_batches
        self.negative_rate = negative_rate
        self.report_interval=report_interval
        self.cuda = cuda
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        self.train_data_raw = None
        self.valid_data_raw = None
        self.test_data_raw = None
        self.max_epochs = max_epochs
    def process_data(self, separate_relations=False):
        assert self.train_data_raw is not None and self.valid_data_raw is not None and self.test_data_raw is not None
        if self.train_data is not None: return # do-not re-load
        self.train_data = Sampler(self.n_entities, self.train_data_raw, n_batches=self.n_batches, negative_rate=self.negative_rate, report_interval=self.report_interval, epochs=self.max_epochs, separate_relations=separate_relations)
        self.valid_data = Sampler(self.n_entities, self.valid_data_raw, n_batches=self.n_val_batches, negative_rate=self.negative_rate, report_interval=self.report_interval, epochs=self.max_epochs, separate_relations=separate_relations)
        self.test_data = Sampler(self.n_entities, self.test_data_raw, n_batches=self.n_test_batches, negative_rate=self.negative_rate, report_interval=self.report_interval, epochs=self.max_epochs, separate_relations=separate_relations)
    def eval_metrics(self, embeddings, sampler, thresholds=None):
        batches = sampler.batch_generator()
        # get the results separated by relations
        all_scores, all_labels = self.model.calc_score_by_relation(batches, embeddings, cuda=self.cuda)
        # roc --- AUC (curved-based method)
        roc_auc, roc_auc_each = roc_auc_score(all_scores, all_labels)
        pr_auc, pr_auc_each = precision_recall_score(all_scores, all_labels)
        roc_auc_dict = dict(zip(["roc_auc_{}".format(r) for r in self.model.relation_names], roc_auc_each))
        pr_auc_dict = dict(zip(["pr_auc_{}".format(r) for r in self.model.relation_names], pr_auc_each))
        # acc --- based on a selected threshold (threshold-based method)
        if thresholds is None:
            acc, thresholds = link_accuracy(all_scores, all_labels)
        else:
            acc = link_accuracy_fixed_thresholds(all_scores, all_labels, thresholds)
        # hit rate & MRR --- probably be implemented in the future for comparision to other dataset
        ret_dict = {"roc_auc": roc_auc.item(), "pr_auc":pr_auc.item(), "accuracy": acc.item(), "return": {"thresholds":thresholds}}
        ret_dict.update(roc_auc_dict)
        ret_dict.update(pr_auc_dict)
        return ret_dict
    def loss_epoch(self, embeddings):
        loss_train_sum = 0
        sampler = self.train_data
        # do the sampling (from given positive samples) and negative
        batches = sampler.batch_generator()
        for batch_id, triplets, labels, _, _ in batches:
            labels = torch.from_numpy(labels)
            triplets = torch.from_numpy(triplets)
            if self.cuda:
                triplets, labels = triplets.cuda(0), labels.cuda(0)
            loss_train = self.model.get_loss(embeddings, labels, triplets)
            loss_train_sum += loss_train
        return loss_train_sum

class LinkPred_BatchTask(LinkPredictionTask):
    def get_adjs(self, mask_info):
        return self.adjs
    def loss_epoch(self):
        loss_train_sum = 0
        sampler = self.train_data
        # do the sampling (from given positive samples) and negative
        batches = sampler.batch_generator()
        for batch_id, triplets, labels, _, mask_info in batches:
            labels = torch.from_numpy(labels)
            triplets = torch.from_numpy(triplets)
            if self.cuda:
                triplets, labels = triplets.cuda(0), labels.cuda(0)
            self.optimizer.zero_grad()
            features = self.features_generator()
            masked_adjs = self.get_adjs(mask_info)
            embeddings = self.model(features, masked_adjs)
            loss_train = self.model.get_loss(embeddings, labels, triplets)
            loss_train.backward()
            self.optimizer.step()
            loss_train_sum += loss_train.item()
        return loss_train_sum 
    def train_epoch(self):
        self.model.train()
        loss_train = self.loss_epoch()
        return {"loss_train": loss_train}

class TIMMEManager(LinkPredictionTask):
    """
    Task manager for all the TIMME models.
    """
    def __init__(self, model, features_generator, adjs, lr, weight_decay, algorithm="Adam", fastmode=False, lr_scheduler="Step", min_lr=1e-5, epochs=600, n_batches=10, n_val_batches=1, n_test_batches=1, negative_rate = 1.5, cuda=False, report_interval=0, max_epochs=100):
        super().__init__(model, features_generator, adjs, lr, weight_decay, algorithm, fastmode, lr_scheduler, min_lr, epochs, n_batches, n_val_batches, n_test_batches, negative_rate, cuda, report_interval, max_epochs)
        self.task_name = "TIMME "
    def get_adjs(self, mask_info):
        return self.adjs
    def load_data(self, train_data, valid_data, test_data, labels, train_link, valid_link, test_link):
        super().load_data(train_link, valid_link, test_link)
        super().process_data(separate_relations=True)
        self.train_data = (self.train_data, train_data)
        self.valid_data = (self.valid_data, valid_data)
        self.test_data  = (self.test_data,  test_data)
        self.labels = labels
    def loss_epoch(self):
        loss_train_sum = 0
        sampler, train_class_data = self.train_data
        # do the sampling (from given positive samples) and negative
        batches = sampler.batch_generator()
        for batch_id, triplets, labels, _, mask_info in batches:
            if self.cuda:
                labels = [torch.from_numpy(l).cuda(0) for l in labels]
                triplets = [torch.from_numpy(t).cuda(0) for t in triplets]
            else:
                labels = [torch.from_numpy(l) for l in labels]
                triplets = [torch.from_numpy(t) for t in triplets]
            self.optimizer.zero_grad()
            features = self.features_generator()
            masked_adjs = self.get_adjs(mask_info)
            embeddings = self.model(features, masked_adjs) # R+1: link-pred * #link-types, node-classification
            loss_train = self.model.get_loss(embeddings, labels, triplets, mask_info, train_class_data, self.labels)
            loss_train.backward()
            self.optimizer.step()
            loss_train_sum += loss_train.item()
        return loss_train_sum 
    def train_epoch(self):
        self.model.train()
        loss_train = self.loss_epoch()
        return {"loss_train": loss_train}
    def eval_metrics(self, embeddings, data):
        sampler, class_data = data
        batches = sampler.batch_generator()
        # get the results separated by relations
        all_scores, all_labels, _ = self.model.calc_score_by_relation(batches, embeddings, cuda=self.cuda)
        # roc --- AUC (curved-based method)
        _, roc_auc_each = roc_auc_score(all_scores, all_labels)
        _, pr_auc_each = precision_recall_score(all_scores, all_labels)
        roc_auc_dict = dict(zip(["roc_auc_{}".format(r) for r in self.model.relation_names], roc_auc_each))
        pr_auc_dict = dict(zip(["pr_auc_{}".format(r) for r in self.model.relation_names], pr_auc_each))

        # the classification task
        acc_val = accuracy(embeddings[-1][class_data].cpu(), self.labels[class_data].cpu())
        f1_val = f1_score(embeddings[-1][class_data].cpu(), self.labels[class_data].cpu())

        # return the metrics
        ret_dict = {"classification accuracy": acc_val.item(), "classification f1-score": f1_val.item()}
        ret_dict.update(roc_auc_dict)
        ret_dict.update(pr_auc_dict)
        return ret_dict
    def get_pred(self):
        self.model.eval()
        features = self.features_generator()
        embeddings = self.model(features, self.adjs)
        # the node output
        node_output = embeddings[-1].cpu().detach().numpy() # as N * 2 numpy
        # the link output
        sampler, _ = self.test_data
        batches = sampler.batch_generator()
        # get the results separated by relations
        all_scores, all_labels, (all_from, all_to) = self.model.calc_score_by_relation(batches, embeddings[:-1], cuda=self.cuda, get_triplets=True)
        return node_output, (all_scores, all_labels, all_from, all_to)



        
