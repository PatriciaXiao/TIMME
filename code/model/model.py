from model.layer import GraphConvolution
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from model.attention import MultiheadAttention, ScaledDotProductSelfAttention

import torch

from utils import slicing
import numpy as np

class GCN_multirelation(nn.Module):
    """
    The multi-relational encoder of TIMME
    """
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode="none", attention_mode="none"):
        super(GCN_multirelation, self).__init__()

        self.gc1 = GraphConvolution(num_relation, num_entities, num_adjs, nfeat, nhid, attention_mode=attention_mode)
        self.gc2 = GraphConvolution(num_relation, num_entities, num_adjs, self.gc1.out_features, nhid, attention_mode=attention_mode)
        self.dropout = dropout
        if skip_mode not in ["add", "concat", "none"]:
            print("skip mode {} unknown, use default option 'none'".format(skip_mode))
            skip_mode = "add"
        elif skip_mode in ["concat"]:
            self.ff = nn.Linear(self.gc1.out_features + self.gc2.out_features, self.gc2.out_features)
        self.skip_mode = skip_mode
        self.out_dim = self.gc2.out_features

    def skip_connect_out(self, x2, x1):
        return self.ff(torch.cat((x2, x1), 1)) if self.skip_mode=="concat" else x2+x1

    def forward(self, x, adjs):
        x1 = F.relu(self.gc1(x, adjs))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, adjs))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        return x2 if self.skip_mode is "none" else self.skip_connect_out(x2, x1)

class Classification(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, regularization=None, gcn=None, skip_mode="none", attention_mode="none", trainable_features=None):
        super(Classification, self).__init__()
        self.gcn = GCN_multirelation(num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode=skip_mode, attention_mode=attention_mode) if gcn is None else gcn
        self.classifier = nn.Linear(self.gcn.out_dim, nclass)
        self.reg_param = regularization if regularization else 0
        self.trainable_features = trainable_features if trainable_features else None

    def forward(self, x, adjs, calc_gcn=True):
        x = self.gcn(x, adjs) if calc_gcn else x
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

    def regularization_loss(self, embedding):
        if not self.reg_param:
            return 0
        return self.reg_param * torch.mean(embedding.pow(2))

    def get_loss(self, output, labels, idx_lst):
        reg_loss = self.regularization_loss(output) # regularize the embeddings
        return F.nll_loss(output[idx_lst], labels[idx_lst]) + reg_loss

class LinkPrediction(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, dropout, relations=None, regularization=None, gcn=None, skip_mode="none", attention_mode="none", weightless=False, add_layer=True, trainable_features=None):
        super(LinkPrediction, self).__init__()

        self.gcn = GCN_multirelation(num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode=skip_mode, attention_mode=attention_mode) if gcn is None else gcn
        self.trainable_features = trainable_features if trainable_features else None
        if add_layer:
            self.additional_layer = nn.Linear(self.gcn.out_dim, self.gcn.out_dim) 
        else:
            self.register_parameter('additional_layer', None)
        self.reg_param = regularization if regularization else 0
        self.num_relation = num_relation
        self.relation_names = relations if relations else [""] * num_relation
        # relations to predict using weight: could be 1 ~ N relations when we use DistMult
        # each relation's embedding is trained differently anyway
        self.n_relations = num_relation
        self.w_relation = nn.Parameter(torch.Tensor(num_relation, self.gcn.out_dim), requires_grad=(not weightless))
        self.w_standard = nn.Parameter(torch.Tensor(num_relation, self.gcn.out_dim * 2), requires_grad=(not weightless))
        self.bias = nn.Parameter(torch.Tensor(num_relation,1), requires_grad=(not weightless))
        # initialization wouldn't affect if it is trainable or not
        # relations have to be somewhat different from each other to make a difference
        nn.init.xavier_uniform_(self.w_relation,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_standard,
                            gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.bias,
                            gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        '''
        NTN with diag-weight and k=1
        Called TIMME-NTN for convenience in our paper
        '''
        # tensor layer with k = 1 and w being diagonal
        s = embedding[triplets[0]]
        r = self.w_relation[triplets[1]]
        o = embedding[triplets[2]]
        # standard layer 
        v = self.w_standard[triplets[1]]
        c = torch.cat([s,o], dim=1)  # concatenation
        # bias term
        b = self.bias[triplets[1]]
        # final score
        score = torch.sum(s * r * o, dim=1) + torch.sum(v * c, dim=1) + torch.sum(b, dim=1)
        return score

    def forward(self, x, adjs, calc_gcn=True):
        '''
        forward without calculating loss
        '''
        embeddings = self.gcn(x, adjs) if calc_gcn else x
        if self.additional_layer:
            embeddings = self.additional_layer(embeddings)
        return embeddings

    def regularization_loss(self, embedding):
        if not self.reg_param:
            return 0
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embeddings, labels, triplets):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets consists of [source, relation, destination]
        # embeddings = self.forward(x, adjs)
        score = self.calc_score(embeddings, triplets)
        reg_loss = self.regularization_loss(embeddings)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        loss = predict_loss + self.reg_param * reg_loss
        return loss

    def calc_score_by_relation(self, batches, embeddings, cuda=False):
        '''
        batches is a batch generator, from sampler
        see examples in training and testing script
        embeddings is the embedding result of the nodes
        '''
        all_scores = [list() for i in range(self.num_relation)]
        all_labels = [list() for i in range(self.num_relation)]
        for batch_id, triplets, labels, relation_indexes, _ in batches:
            triplets = torch.from_numpy(triplets)
            if cuda:
                triplets = triplets.cuda()
            scores = self.calc_score(embeddings, triplets).detach().cpu().numpy()
            # print(slicing(triplets.numpy().transpose(), relation_indexes[1]))
            for r in range(self.n_relations):
                score_r = slicing(scores, relation_indexes[r])
                label_r = slicing(labels, relation_indexes[r])
                all_scores[r].append(score_r)
                all_labels[r].append(label_r)
        # get the scores of different relation and their labels
        all_scores = [np.concatenate(scores_r) for scores_r in all_scores]
        all_labels = [np.concatenate(labels_r) for labels_r in all_labels]
        return all_scores, all_labels

class TIMME(nn.Module):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, regularization=None, skip_mode="none", attention_mode="none",trainable_features=None):
        super(TIMME, self).__init__()
        self.gcn = GCN_multirelation(num_relation, num_entities, num_adjs, nfeat, nhid, dropout, skip_mode=skip_mode, attention_mode=attention_mode)
        self.trainable_features = trainable_features if trainable_features else None
        # the last model is always node classification, following the R relations samples
        self.models = nn.ModuleList(list())
        self.num_relation = num_relation
        self.relation_names = relations
        # treat each relation separately
        for i in range(num_relation):
            self.models.append(LinkPrediction(1, num_entities, num_adjs, nfeat, nhid, dropout, regularization=regularization, gcn=self.gcn))
        self.models.append(Classification(num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, gcn=self.gcn))
    def forward(self, x, adjs):
        gcn_embedding = self.gcn(x, adjs)
        return [m(gcn_embedding, adjs, calc_gcn=False) for m in self.models]
    def calc_joint_loss(self, embeddings, losses):
        # no lambda here
        return sum(losses)

    def get_loss(self, embeddings, labels, triplets, mask_info, class_index, class_labels):
        link_loss = [self.models[i].get_loss(embeddings[i], labels[i], triplets[i]) for i in range(self.num_relation)]
        mask_idxs = set(np.concatenate(np.array(np.concatenate(mask_info))))
        valid_idxs = set(class_index.tolist()).intersection(mask_idxs)
        valid_idx_idxs = [i for i,idx in enumerate(class_index.tolist()) if idx in valid_idxs]
        class_index = class_index[valid_idx_idxs]
        node_loss = self.models[-1].get_loss(embeddings[-1], class_labels, class_index)
        # calculate the loss
        return self.calc_joint_loss(embeddings[:-1], link_loss + [node_loss])
            
    def calc_score_by_relation(self, batches, embeddings, cuda=False, get_triplets=False):
        '''
        batches is a batch generator, from sampler
        see examples in training and testing script
        embeddings is the embedding result of the nodes
        '''
        all_scores = [list() for i in range(self.num_relation)]
        all_labels = [list() for i in range(self.num_relation)]
        all_from = [list() for i in range(self.num_relation)] if get_triplets else None
        all_to = [list() for i in range(self.num_relation)] if get_triplets else None
        for batch_id, triplets, labels, relation_indexes, _ in batches:
            if cuda:
                triplets = [torch.from_numpy(t).cuda(0) for t in triplets]
            else:
                triplets = [torch.from_numpy(t) for t in triplets]
            scores = [self.models[i].calc_score(embeddings[i], triplets[i]).detach().cpu().numpy() for i in range(self.num_relation)]
            for r in range(self.num_relation):
                all_scores[r].append(scores[r][:])
                all_labels[r].append(labels[r][:])
                if get_triplets:
                    all_from[r].extend(list(triplets[r][0].numpy()))
                    all_to[r].extend(list(triplets[r][2].numpy()))
        # get the scores of different relation and their labels
        all_scores = [np.concatenate(scores_r) for scores_r in all_scores]
        all_labels = [np.concatenate(labels_r) for labels_r in all_labels]
        all_triplets = (all_from, all_to)
        return all_scores, all_labels, all_triplets

class TIMMEhierarchical(TIMME):
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, regularization=None, skip_mode="none", attention_mode="none", trainable_features=None):
        super(TIMMEhierarchical, self).__init__(num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, regularization=regularization, skip_mode=skip_mode, attention_mode=attention_mode,trainable_features=trainable_features)
        self._lambda = ScaledDotProductSelfAttention(nhid, num_entities) 
        self.attention_weight = None
    def forward(self, x, adjs):
        gcn_embedding = self.gcn(x, adjs)
        link_embeddings = [m(gcn_embedding, adjs, calc_gcn=False) for m in self.models[:-1]]
        attention_weight = self._lambda(torch.stack(link_embeddings))
        node_x_in = torch.sum(attention_weight * torch.stack(link_embeddings, 2), 2)
        node_embedding = self.models[-1](node_x_in, adjs, calc_gcn=False)
        self.attention_weight = attention_weight.detach().numpy()
        return link_embeddings + [node_embedding]
    
class TIMMEsingle(TIMME):
    """
    The variation that uses only a single relation's data for training.
    Useful as baseline.
    """
    def __init__(self, num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, regularization=None, skip_mode="none", attention_mode="none", trainable_features=None, relation_id=0):
        super(TIMMEsingle, self).__init__(num_relation, num_entities, num_adjs, nfeat, nhid, nclass, dropout, relations, regularization=regularization, skip_mode=skip_mode, attention_mode=attention_mode,trainable_features=trainable_features)
        self.relation_id = relation_id

    def calc_joint_loss(self, embeddings, losses):
        return losses[self.relation_id]



