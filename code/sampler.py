import numpy as np
import multiprocessing as mp
from utils import flatten, pair_set, from_list, to_list

class NaiveSampler:
    def __init__(self, n_entities, link_info, n_batches=10, negative_rate=1.5, report_interval=0, epochs=100, separate_relations=False):
        # raw data
        self.n_relations = len(link_info)
        self.n_entities = n_entities
        self.n_batches = n_batches
        self.link_info = link_info
        self.negative_rate = negative_rate
        self.report_interval = report_interval
        self.max_epochs = epochs
        self.separate_relations = separate_relations
        
        # the information to be directly used for generating batches
        all_index_list = np.array(range(n_entities))
        positive_sizes = np.ceil(np.array([r_links.shape[1] for r_links in link_info]) / n_batches).astype(int).reshape(1,-1)
        negative_sizes = (positive_sizes * negative_rate).astype(int).reshape(1,-1) # the negative size per batch for each relation
        self.batch_sizes = np.concatenate((negative_sizes, positive_sizes))
        # localize the sampling to accelerate the process --- inspired by GraphVite
        self.known_positive_links = [pair_set(r_links[0], r_links[1]) for r_links in link_info]
        self.self_loop_links = pair_set(all_index_list, all_index_list)

    def single_rel_sampler(self, batch_id, relation_id, link_info, pos_neg=1, remove_false_neg=True):
        '''
        it is okay not to remove the "false negative", for it is not really a large portion
        '''
        batch_size = self.batch_sizes[pos_neg][relation_id]
        if pos_neg: 
            # pos_neg is 1, positibe sampling
            positive_samples = self.link_info[relation_id][:2, batch_id*batch_size : (batch_id+1)*batch_size]
            samples_from = positive_samples[0, :]
            samples_to   = positive_samples[1, :]
        else:
            # pos_neg is 0, negative sampling
            known_set = self.known_positive_links[relation_id]
            samples_from = np.random.choice(self.n_entities, batch_size)
            samples_to = np.random.choice(self.n_entities, batch_size)
            if remove_false_neg:
                # get the pairs as set and remove "false negatives" (those that are positive)
                raw_negative_pair_set = pair_set(list(samples_from), list(samples_to))
                filtered_negative_pairs = list(raw_negative_pair_set - known_set - self.self_loop_links)
                # extract the from and to infomation
                samples_from = from_list(filtered_negative_pairs)
                samples_to = to_list(filtered_negative_pairs)
        i_label = 0 if self.separate_relations else relation_id
        samples_relation = [i_label] * len(samples_from)
        return samples_from, samples_relation, samples_to

    def pos_neg_sampler(self, batch_id=None, pos_neg=1):
        return_holder = [samples_from, samples_relation, samples_to] = [list(), list(), list()]
        for i in range(self.n_relations): # i is relation_id
            single_results = self.single_rel_sampler(batch_id, i, self.link_info[i], pos_neg)
            for j,tmp_result in enumerate(single_results):
                return_holder[j].append(tmp_result)
        if self.separate_relations:
            all_samples, all_labels = list(), list()
            for from_list,r_list,to_list in zip(samples_from, samples_relation, samples_to):
                all_samples.append(np.array([from_list, r_list, to_list]))
                all_labels.append(np.ones(all_samples[-1].shape[1]).astype(np.float32) * pos_neg)
        else:
            all_samples = np.array([flatten(samples_from), flatten(samples_relation), flatten(samples_to)])
            all_labels = np.ones(all_samples.shape[1]).astype(np.float32) * pos_neg
        from_to = (samples_from, samples_to)
        return all_samples, all_labels, from_to

    def get_relation_index_slice(self, positive_relation_length, negative_relation_length):
        relation_indexes = list()
        positive_cursor = 0
        negative_cursor = sum(positive_relation_length)
        for i in range(self.n_relations):
            relation_indexes.append([[positive_cursor, positive_cursor + positive_relation_length[i]], \
                                     [negative_cursor, negative_cursor + negative_relation_length[i]]])
            positive_cursor += positive_relation_length[i]
            negative_cursor += negative_relation_length[i]
        return relation_indexes

    def get_next_positive(self, batch_id):
        return self.pos_neg_sampler(batch_id)
    def get_next_negative(self):
        return self.pos_neg_sampler(pos_neg=0)

    def batch_generator(self):
        '''
        the generator could be called by next(XXX) or iteration
        '''
        for batch_id in range(self.n_batches):
            # decide whether or not we should report in the current batch
            batch_report = self.report_interval > 0 and ((batch_id + 1) % self.report_interval == 0)

            if batch_report: print("batch ({}/{})\tstart positive sampling".format(batch_id+1, self.n_batches))
            all_positive_samples, all_positive_labels, mask_info = self.get_next_positive(batch_id)

            if batch_report: print("\tstart negative sampling")
            # accelerate negative sampling by localizing the sampling --- check for only single relation
            all_negative_samples, all_negative_labels, mask_info_neg = self.get_next_negative()

            if batch_report: print("\tgeting the labels and triplets")
            if self.separate_relations:
                triplets = [np.concatenate((ps, ns), axis=1) for ps,ns in zip(all_positive_samples, all_negative_samples)] 
                labels = [np.concatenate((pl, nl)) for pl, nl in zip(all_positive_labels, all_negative_labels)]
                relation_indexes = None
            else:
                triplets = np.concatenate((all_positive_samples, all_negative_samples), axis=1)
                labels = np.concatenate((all_positive_labels, all_negative_labels)) #.reshape(1,-1)

                if batch_report: print("\tgeting the indexes of samples of each relation (used for calculating loss)")
                positive_relation_length = [len(x) for x in mask_info[0]]
                negative_relation_length = [len(x) for x in mask_info_neg[0]]
                relation_indexes = self.get_relation_index_slice(positive_relation_length, negative_relation_length)

            yield batch_id, triplets, labels, relation_indexes, mask_info

class MultiprocessSampler(NaiveSampler):
    '''
    https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/sequentialRec/markovChains/sampler.py
    https://docs.python.org/2/library/multiprocessing.html
    * using a Queue for cached batches
    * using mp.Process for multi-processing (don't use Pool, that's too slow)
    in theory it should work, but our problem is that we did some optimization on the sampler so that it runs really fast
        and thus when doing this multiprocess thing it keeps pushing everything into the queue and there comes memory issue
        memory issue, because exceeded the maximum length of the queue
    in this sense, we actually argue that we do not really need multi-processing
    restricting the number of epochs, we observe only around 5 seconds benefit (out of around 30) each epoch using multi-process
    '''
    def __init__(self, n_entities, link_info, n_batches=10, negative_rate=1.5, report_interval=0, epochs=100, maxsize=32767, n_workers=10):
        '''
        multi-process queue allows at most 32767 length
        '''
        super().__init__(n_entities, link_info, n_batches, negative_rate, report_interval, epochs)
        self.maxsize = maxsize
        self.n_workers = n_workers
        # positive sampling could not be faster, thus it must be negative sampling that could do something to accelerate
        self.neg_queue = mp.Queue(maxsize=self.maxsize)
        # the processors
        batches_each_processor = np.ceil(self.n_batches / n_workers).astype(int) # ceil
        self.processors = [ mp.Process(target=self.negative_sampler, args=(batches_each_processor,)) \
                            for i in range(n_workers) ]
        for p in self.processors:
            p.daemon = True
            p.start()
    def __del__(self):
        '''
        the destructor we need for ending the processes
        automatically called when the whole program ends
        '''
        for p in self.processors:
            p.terminate()
            p.join()
    def negative_sampler(self, n_batches):
        # while True: # wouldn't work well on laptop where GPU is not available and everyone competes for CPU
        for e in range(self.max_epochs):
            # each epoch
            for i in range(n_batches):
                next_batch_neg = self.pos_neg_sampler(pos_neg=0)
                self.neg_queue.put(next_batch_neg)
            # time.sleep(20)
    def get_next_negative(self):
        try:
            if not self.neg_queue.empty(): # note: this function is not 100% reliable
                return self.neg_queue.get(timeout=0.5)
            else:
                return self.pos_neg_sampler(pos_neg=0)
        except:
            # if it is empty
            return self.pos_neg_sampler(pos_neg=0)



