from __future__ import absolute_import, division, print_function

import random
from collections import defaultdict

from tqdm import tqdm

from preprocessor import textProcess
import argparse
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter
import faiss
from faiss import normalize_L2
import scipy
import torch.nn.functional as F

# Default Parameters
epoch = 10
eval_batch_size = 16
train_batch_size = 8
seed_num = 42

flag_train = True
flag_test = True
key = ''
pro = ''

# Important Dir
result_path = 'results'  # your result path
model_output_path = 'saved_models'  # your saved model path
data_dir = 'path to dataset'  # your path to dataset dir

# model path
text_model_path = 'roberta-large'
code_model_path = 'microsoft/codebert-base'


def set_seed(seed=42):
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 issue_key,
                 commit_sha,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label
        self.issue_key = issue_key,
        self.commit_sha = commit_sha,


def MySubSampler(df, x):
    X, y = df[['Issue_KEY', 'Commit_SHA', 'Issue_Text', 'Commit_Text', 'Commit_Code']], df['label']
    rus = RandomUnderSampler(random_state=x)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    return df.sample(frac=1)


def getargs():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=data_dir, type=str,
                        help="data_dir")
    parser.add_argument("--output_dir", default=model_output_path, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default=result_path, type=str,
                        help="The output directory where the result files will be written.")

    # Other parameters
    parser.add_argument("--text_model_path", default=text_model_path, type=str,
                        help="The NL-NL model checkpoint for weights initialization.")
    parser.add_argument("--code_model_path", default=code_model_path, type=str,
                        help="The NL-PL model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default=text_model_path, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=flag_train, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=flag_test, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=eval_batch_size, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=seed_num,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=epoch,
                        help="num_train_epochs")
    parser.add_argument("--key", default=key, type=str,
                        help="Key of the project.")
    parser.add_argument("--pro", default=pro, type=str,
                        help="The used project.")
    parser.add_argument("--num_class", default=2, type=int,
                        help="The number of classes.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    return args


def convert_examples_to_features(row, tokenizer, commitType, args):
    if 'Token' in commitType:
        issue_token = row['Issue_Text_Token']
        commit_token = row[commitType]
    else:
        issue_text = textProcess(row['Issue_Text'], args.key)
        commit_text = textProcess(row[commitType], args.key)
        issue_token = tokenizer.tokenize(issue_text)
        commit_token = tokenizer.tokenize(commit_text)
    if len(issue_token) + len(commit_token) > args.max_seq_length - 3:
        if len(issue_token) > (args.max_seq_length - 3) / 2 and len(commit_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:int((args.max_seq_length - 3) / 2)]
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
        elif len(issue_token) > (args.max_seq_length - 3) / 2:
            issue_token = issue_token[:args.max_seq_length - 3 - len(commit_token)]
        elif len(commit_token) > (args.max_seq_length - 3) / 2:
            commit_token = commit_token[:args.max_seq_length - 3 - len(issue_token)]
    combined_token = [tokenizer.cls_token] + issue_token + [tokenizer.sep_token] + commit_token + [tokenizer.sep_token]
    combined_ids = tokenizer.convert_tokens_to_ids(combined_token)
    if len(combined_ids) < args.max_seq_length:
        padding_length = args.max_seq_length - len(combined_ids)
        combined_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(combined_token, combined_ids, row['label'], row['Issue_KEY'], row['Commit_SHA'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.text_examples = []
        self.code_examples = []
        if 'TRAIN' in file_path:
            df_link = MySubSampler(pd.read_csv(file_path), args.seed)
        else:
            df_link = pd.read_csv(file_path)
        # token + id + label
        bar = tqdm(initial=0, total=df_link.shape[0], desc='generate_features')
        for i_row, row in df_link.iterrows():
            self.text_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Text', args))
            self.code_examples.append(convert_examples_to_features(row, tokenizer, 'Commit_Code', args))
            bar.update()
        bar.close()
        assert len(self.text_examples) == len(self.code_examples), 'ErrorLength'

    def __len__(self):
        return len(self.text_examples)

    def __getitem__(self, i):
        return (torch.tensor(self.text_examples[i].input_ids),
                torch.tensor(self.code_examples[i].input_ids), torch.tensor(self.text_examples[i].label),
                self.text_examples[i].issue_key, self.text_examples[i].commit_sha)


class ArtifactsDataset(Dataset):
    def __init__(self, text_examples, code_examples):
        self.text_examples = text_examples
        self.code_examples = code_examples

    def __len__(self):
        return len(self.text_examples)

    def __getitem__(self, i):
        return (torch.tensor(self.text_examples[i].input_ids),
                torch.tensor(self.code_examples[i].input_ids), torch.tensor(self.text_examples[i].label),
                self.text_examples[i].issue_key, self.text_examples[i].commit_sha)


class UnlabArtifactsDataset(Dataset):
    def __init__(self, text_examples, code_examples):
        self.text_examples = text_examples
        self.code_examples = code_examples

    def __len__(self):
        return len(self.text_examples)

    def __getitem__(self, i):

        return (torch.tensor(self.text_examples[i][0].input_ids),
                torch.tensor(self.code_examples[i][0].input_ids),
                torch.tensor(self.text_examples[i][0].label),
                self.text_examples[i][1],
                self.text_examples[i][0].issue_key, self.text_examples[i][0].commit_sha)


class Artifacts:
    def __init__(self, tokenizer, args, file_path):
        self.tokenizer = tokenizer
        self.args = args
        self.df = pd.read_csv(file_path)
        self.iss_data = dict()
        self.comm_data = dict()
        self.rel_index = defaultdict(set)
        bar = tqdm(initial=0, total=self.df.shape[0], desc='artifacts processing')
        for i_row, row in self.df.iterrows():
            iss_id = row['Issue_KEY']
            comm_id = row['Commit_SHA']
            label = row['label']
            iss_text = row['Issue_Text']
            comm_text = row['Commit_Text']
            comm_code = row['Commit_Code']

            # iss_text_token = tokenizer.tokenize(textProcess(iss_text, args.key))
            # comm_text_token = tokenizer.tokenize(textProcess(comm_text, args.key))
            # comm_code_token = tokenizer.tokenize(textProcess(comm_code, args.key))

            if iss_id not in self.iss_data:
                self.iss_data[iss_id] = {
                    'text': iss_text,
                    'token': tokenizer.tokenize(textProcess(iss_text, args.key)),
                }
            if comm_id not in self.comm_data:
                self.comm_data[comm_id] = {
                    'text': comm_text,
                    't_token': tokenizer.tokenize(textProcess(comm_text, args.key)),
                    'code': comm_code,
                    'c_token': tokenizer.tokenize(textProcess(comm_code, args.key)),
                }

            if label == 1:
                self.rel_index[iss_id].add(comm_id)
            bar.update()
        bar.close()

    def generate_features(self, pairs):
        data = [[label, iss_id, comm_id, self.iss_data[iss_id]['token'], self.comm_data[comm_id]['t_token'], self.comm_data[comm_id]['c_token']] for iss_id, comm_id, label in pairs]
        df = pd.DataFrame(data, columns=['label', 'Issue_KEY', 'Commit_SHA', 'Issue_Text_Token', 'Commit_Text_Token', 'Commit_Code_Token'], dtype=object)
        text_examples = []
        code_examples = []
        # token + id + label
        bar = tqdm(initial=0, total=df.shape[0], desc='generate_features')
        for i_row, row in df.iterrows():
            text_examples.append(convert_examples_to_features(row, self.tokenizer, 'Commit_Text_Token', self.args))
            code_examples.append(convert_examples_to_features(row, self.tokenizer, 'Commit_Code_Token', self.args))
            bar.update()
        bar.close()
        assert len(text_examples) == len(code_examples), 'ErrorLength'
        return text_examples, code_examples

    def generate_unlab_features(self, pairs, need_num):
        data = [[label, iss_id, comm_id, self.iss_data[iss_id]['token'], self.comm_data[comm_id]['t_token'], self.comm_data[comm_id]['c_token'], weight] for iss_id, comm_id, label, weight in pairs]
        df = pd.DataFrame(data, columns=['label', 'Issue_KEY', 'Commit_SHA', 'Issue_Text_Token', 'Commit_Text_Token', 'Commit_Code_Token', 'weight'], dtype=object)
        label1_df = df[df.label==1].sort_values(by='weight',ascending=False).reset_index(drop=True).iloc[:need_num//2]
        label0_df = df[df.label==0].sort_values(by='weight',ascending=False).reset_index(drop=True).iloc[:need_num//2]
        df = pd.concat([label1_df, label0_df], axis=0).reset_index(drop=True)
        text_examples = []
        code_examples = []
        # token + id + label
        bar = tqdm(initial=0, total=df.shape[0], desc='generate_features')
        for i_row, row in df.iterrows():
            text_examples.append([convert_examples_to_features(row, self.tokenizer, 'Commit_Text_Token', self.args), row['weight']])
            code_examples.append([convert_examples_to_features(row, self.tokenizer, 'Commit_Code_Token', self.args), row['weight']])
            bar.update()
        bar.close()
        assert len(text_examples) == len(code_examples), 'ErrorLength'
        return text_examples, code_examples

    def pair_by_datetime(self):
        unlab_iss_df = self.df[['Issue_KEY', 'creationdate', 'resolutiondate']].copy()
        unlab_iss_df.drop_duplicates('Issue_KEY', inplace=True)
        unlab_iss_df.drop(unlab_iss_df[unlab_iss_df['creationdate'] == '0'].index, inplace=True)
        unlab_iss_df.drop(unlab_iss_df[unlab_iss_df['resolutiondate'] == '0'].index, inplace=True)
        unlab_iss_df = unlab_iss_df.reset_index(drop=True)
        unlab_iss_df['creationdate'] = pd.to_datetime(unlab_iss_df['creationdate'])
        unlab_iss_df['resolutiondate'] = pd.to_datetime(unlab_iss_df['resolutiondate'])
        unlab_comm_df = self.df[['Commit_SHA', 'commitdate']].copy()
        unlab_comm_df.drop_duplicates('Commit_SHA', inplace=True)
        unlab_comm_df.drop(unlab_comm_df[unlab_comm_df['commitdate'] == '0'].index, inplace=True)
        unlab_comm_df = unlab_comm_df.reset_index(drop=True)
        unlab_comm_df['commitdate'] = pd.to_datetime(unlab_comm_df['commitdate'])
        # 按日期规则配对
        datetime_unlab_pairs = []
        for rowi, row in unlab_iss_df.iterrows():
            iss_id = row['Issue_KEY']
            comm_df = unlab_comm_df.loc[(unlab_comm_df['commitdate'] >= row['creationdate']) & (
                        unlab_comm_df['commitdate'] <= row['resolutiondate'])]
            if comm_df.shape[0] <= 10:
                comm_ids = comm_df['Commit_SHA'].tolist()
                for comm_id in comm_ids:
                    label = 1 if self.__is_positive_case(iss_id, comm_id) else 0
                    datetime_unlab_pairs.append((iss_id, comm_id, label))
        print('examples by datatime:', len(datetime_unlab_pairs))
        return datetime_unlab_pairs

    def __is_positive_case(self, iss_id, comm_id):
        if iss_id not in self.rel_index:
            return False
        rel_pls = set(self.rel_index[iss_id])
        return comm_id in rel_pls

    def pair_by_DRNS(self):
        pos, neg = [], []
        for iss_id in self.rel_index:
            pos_comm_ids = self.rel_index[iss_id]
            for comm_id in pos_comm_ids:
                pos.append((iss_id, comm_id, 1))
            sample_num = len(pos_comm_ids)
            sel_neg_ids = self.exclude_and_sample(list(self.comm_data.keys()), pos_comm_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((iss_id, n_id, 0))
        return pos + neg

    def update_plabels(self, plabels_data, p_cutoff):
        pseudo_pos_index = dict()
        pseudo_neg_index = dict()
        for nl_id, pl_id, sim_score, weight, plabel in plabels_data:
            if plabel == 1:
                if sim_score[1] > p_cutoff:
                    pseudo_pos_index.setdefault(nl_id, {})[pl_id] = weight
            else:
                assert plabel == 0
                pseudo_neg_index.setdefault(nl_id, {})[pl_id] = weight
        self.pseudo_pos_index = pseudo_pos_index
        self.pseudo_neg_index = pseudo_neg_index
        return True if len(pseudo_pos_index) > 0 else False

    def pair_by_DRNS_and_plabels(self):
        pos, neg = [], []
        for nl_id in self.pseudo_pos_index:
            pos_pl_ids = self.pseudo_pos_index[nl_id].keys()
            for p_id in pos_pl_ids:
                weight = self.pseudo_pos_index[nl_id][p_id]
                pos.append((nl_id, p_id, 1, weight))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = self.exclude_and_sample(list(self.comm_data.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                weight = 1
                if nl_id in self.pseudo_neg_index:
                    if n_id in self.pseudo_neg_index[nl_id]:
                        weight = self.pseudo_neg_index[nl_id][n_id]
                neg.append((nl_id, n_id, 0, weight))
        return pos + neg

    def exclude_and_sample(self, sample_pool, exclude, num):
        sample_pool_copy = sample_pool.copy()
        for id in exclude:
            sample_pool_copy.remove(id)
        selected = random.sample(list(sample_pool_copy), num)
        return selected


def label_propagation_dataloader(lab_pairs, unlab_pairs, train_lab_dataset, train_unlab_dataset, tokenizer, args):
    train_lab_df = pd.DataFrame(lab_pairs, columns=['nl_id', 'pl_id', 'label'])
    train_unlab_df = pd.DataFrame(unlab_pairs, columns=['nl_id', 'pl_id', 'label'])
    train_lab_df['islab'] = 1
    train_unlab_df['islab'] = 0
    train_df = pd.concat([train_lab_df, train_unlab_df]).reset_index(drop=True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    lab_idxs = train_df[train_df['islab'] == 1].index
    unlab_idxs = train_df[train_df['islab'] == 0].index

    print('start generate label_propagation_data')
    lp_data = label_propagation_data(
        train_df.values.tolist(),
        lab_idxs,
        unlab_idxs,
        tokenizer,
        args,
        train_lab_dataset.iss_data,
        train_lab_dataset.comm_data,
        train_unlab_dataset.iss_data,
        train_unlab_dataset.comm_data,
    )
    text_examples, code_examples = lp_data.generate_features()
    lp_dataset = ArtifactsDataset(text_examples, code_examples)
    lp_dataLoader = torch.utils.data.DataLoader(lp_dataset, batch_size=args.train_batch_size, shuffle=False, pin_memory=True, drop_last=False)

    return lp_dataLoader, lp_data


class label_propagation_data:
    def __init__(self, links, lab_idxs, unlab_idxs, tokenizer, args, lab_iss_data, lab_comm_data, unlab_iss_data, unlab_comm_data):
        self.links = links
        self.lab_idxs = lab_idxs
        self.unlab_idxs = unlab_idxs
        self.labels = [link[2] for link in self.links]
        # self.num_classes = len(Counter(self.labels).keys())
        self.num_classes = 2

        self.tokenizer = tokenizer
        self.args = args
        self.lab_iss_data = lab_iss_data
        self.lab_comm_data = lab_comm_data
        self.unlab_iss_data = unlab_iss_data
        self.unlab_comm_data = unlab_comm_data

    def generate_features(self):
        data = []
        for iss_id, comm_id, label, islab in self.links:
            if islab == 1:
                iss_text_token = self.lab_iss_data[iss_id]['token']
                comm_text_token = self.lab_comm_data[comm_id]['t_token']
                comm_code_token = self.lab_comm_data[comm_id]['c_token']
            else:
                assert islab == 0
                iss_text_token = self.unlab_iss_data[iss_id]['token']
                comm_text_token = self.unlab_comm_data[comm_id]['t_token']
                comm_code_token = self.unlab_comm_data[comm_id]['c_token']
            data.append([label, iss_id, comm_id, iss_text_token, comm_text_token, comm_code_token])
        df = pd.DataFrame(data, columns=['label', 'Issue_KEY', 'Commit_SHA', 'Issue_Text_Token', 'Commit_Text_Token', 'Commit_Code_Token'], dtype=object)
        text_examples = []
        code_examples = []
        # token + id + label
        bar = tqdm(initial=0, total=df.shape[0], desc='generate_features')
        for i_row, row in df.iterrows():
            text_examples.append(convert_examples_to_features(row, self.tokenizer, 'Commit_Text_Token', self.args))
            code_examples.append(convert_examples_to_features(row, self.tokenizer, 'Commit_Code_Token', self.args))
            bar.update()
        bar.close()
        assert len(text_examples) == len(code_examples), 'ErrorLength'
        return text_examples, code_examples

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

        D, I = index.search(X, k + 1)

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

        plabels_0 = list()
        plabels_1 = list()
        res = list()
        for idx in unlabeled_idx:
            nl_id, pl_id, label, islab = self.links[idx]
            assert islab == 0
            sim_score = probs_l1[idx]
            weight = weights[idx]
            p_label = p_labels[idx]
            res.append([nl_id, pl_id, sim_score, weight, p_label])
            if label == 0:
                plabels_0.append(p_label)
            else:
                assert label == 1
                plabels_1.append(p_label)
        print(f'label_0 num: {len(plabels_0)}, p_label_0 num: {plabels_0.count(0)}, p_label_1 num: {plabels_0.count(1)}')
        print(f'label_1 num: {len(plabels_1)}, p_label_0 num: {plabels_1.count(0)}, p_label_1 num: {plabels_1.count(1)}')
        return res