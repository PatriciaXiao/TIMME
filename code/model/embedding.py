import torch

class PartlyLearnableEmbedding(object):
    """
    When some features are missing and we want to make these features trainable
    We need to treat the embedding as containing two parts: one known features, one trainable.
    mask is used to determine which portion is trainable, and which portion should be fixed.
    """
    def __init__(self, num_embeddings, features, trainable, mask, cuda=False):
        self.num_embeddings = num_embeddings
        self.feat_index = torch.LongTensor(range(num_embeddings))
        self.features = features
        self.trainable = trainable
        self.mask = mask.cuda(0) if cuda else mask
        self.cuda = cuda
    def get_features(self):
        '''
        features, when trainable, must be generated once again every time we run an epoch
        otherwise, we have to do .backward(retain_graph=True)
        and that could turn out to be memory-consuming
        '''
        f = self.features(self.feat_index)
        if self.cuda:
            f = f.cuda(0)
        f[self.mask] = self.trainable(self.mask)
        # and actually if space permitted, we can store both fixed and trainable embeddings on GPU when using cuda
        return f
    def __call__(self):
        return self.get_features()

class FixedFeature(object):
    """
    The ordinary (dafault) version of feature X
    Known, fixed, not going to be trained at all
    """
    def __init__(self, features, cuda=False):
        if cuda:
            self.features = features.cuda(0)
        else:
            self.features = features
        self.cuda = cuda
    def __call__(self):
        return self.features
