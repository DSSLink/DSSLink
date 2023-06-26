import itertools
import time
import numpy as np
import pandas as pd
from collections import Counter
import faiss
from faiss import normalize_L2
import scipy
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler, SubsetRandomSampler, RandomSampler


def load_unlabeled_data(train_lab_examples, train_unlab_examples, batch_size, pairing_rule):
    train_lab_links = train_lab_examples.random_neg_sampling_dataloader(batch_size=batch_size, data_loader_flag=False)
    if pairing_rule == 'random':
        train_unlab_links = train_unlab_examples.gen_unlab_examples_by_random()
    else:
        train_unlab_links = train_unlab_examples.gen_unlab_examples_by_datetime()

    train_lab_df = pd.DataFrame(train_lab_links, columns=['nl_id', 'pl_id', 'label'])
    train_unlab_df = pd.DataFrame(train_unlab_links, columns=['nl_id', 'pl_id', 'label'])
    train_lab_df['islab'] = 1
    train_unlab_df['islab'] = 0
    train_df = pd.concat([train_lab_df, train_unlab_df]).reset_index(drop=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    lab_idxs = train_df[train_df['islab'] == 1].index
    unlab_idxs = train_df[train_df['islab'] == 0].index

    train_dataset = loader_traindata(train_df.values.tolist(), lab_idxs, unlab_idxs)
    train_loader_noshuff = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       # num_workers=args.workers,  # Needs images twice as fast
                                                       pin_memory=True,
                                                       drop_last=False)

    return train_loader_noshuff, train_dataset, len(train_unlab_links)

class loader_traindata(Dataset):
    def __init__(self, links, lab_idxs, unlab_idxs):
        self.links = links
        self.lab_idxs = lab_idxs
        self.unlab_idxs = unlab_idxs
        self.labels = [link[2] for link in self.links]
        self.num_classes = len(Counter(self.labels).keys())

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        nl_id, pl_id, label, islab = self.links[idx]
        
        if idx in self.lab_idxs:
            assert islab == 1
        else:
            assert idx in self.unlab_idxs
            assert islab == 0

        return nl_id, pl_id, label, islab

    def label_propagation(self, X, k=50, max_iter=20, alpha=0.99):
        print('Updating pseudo-labels...')
        labels = np.asarray(self.labels)
        labeled_idx = np.asarray(self.lab_idxs)
        unlabeled_idx = np.asarray(self.unlab_idxs)

        # kNN search for the graph
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

        normalize_L2(X)
        index.add(X)
        N = X.shape[0]
        Nidx = index.ntotal

        c = time.time()
        D, I = index.search(X, k + 1)
        elapsed = time.time() - c
        print('kNN Search done in %d seconds' % elapsed)

        # Create the graph
        D = D[:, 1:] ** 3
        I = I[:, 1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (k, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N, self.num_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(self.num_classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(self.num_classes)
        weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)

        correct_idx = (p_labels == labels)
        print('selection unlabidx accuracy: %.2f' % correct_idx[unlabeled_idx].mean())

        res = list()
        for idx in unlabeled_idx:
            nl_id, pl_id, label, islab = self.links[idx]
            assert islab == 0
            sim_score = probs_l1[idx]
            weight = weights[idx]
            p_label = p_labels[idx]
            res.append([nl_id, pl_id, sim_score, weight, p_label])
        return res

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)