import os
import random
from collections import defaultdict
from typing import List, Tuple
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, Dataset

from common.models import BertClassifier

logger = logging.getLogger(__name__)

F_ID = 'id'
F_TOKEN = 'tokens'
F_ATTEN_MASK = "attention_mask"
F_INPUT_ID = "input_ids"
F_EMBD = "embd"
F_TK_TYPE = "token_type_ids"


def load_examples(data_dir, model, use_new_links=False):
    logger.info('Creating examples from dataset file at {}'.format(data_dir))
    raw_examples = read_examples(data_dir, use_new_links)
    examples = Examples(raw_examples)
    examples.update_features(model)
    return examples

def read_examples(data_dir, use_new_links=False):
    issues = read_artifacts(os.path.dirname(os.path.dirname(data_dir)), "issue")
    commits = read_artifacts(os.path.dirname(os.path.dirname(data_dir)), "commit")
    links = read_artifacts(data_dir, "link")
    issue_index = {iss_id: iss_text for iss_id, iss_text in issues}
    commit_index = {cm_id: cm_text for cm_id, cm_text in commits}
    if use_new_links:
        logger.info('Use new links')
        new_issues = read_artifacts(os.path.dirname(os.path.dirname(data_dir)), 'new_issue')
        new_commits = read_artifacts(os.path.dirname(os.path.dirname(data_dir)), 'new_commit')
        for iss_id, iss_text in new_issues:
            issue_index[iss_id] = iss_text
        for cm_id, cm_text in new_commits:
            commit_index[cm_id] = cm_text
    examples = []
    for iss_id, cm_id in links:
        example = {
            "NL": [iss_id, issue_index[iss_id]],
            "PL": [cm_id, commit_index[cm_id]]
        }
        examples.append(example)
    return examples

def read_artifacts(dataset_dir, type):
    file_path = os.path.join(dataset_dir, f'{type}.csv')
    df = pd.read_csv(file_path)
    df = df.replace(np.nan, regex=True)
    arti = []
    for index, row in df.iterrows():
        if type == 'issue' or type == 'new_issue':
            iss_id = row['issue_id']
            iss_text = row['issue_desc'] + ' ' + row['issue_comments']
            art = (iss_id, iss_text)
        elif type == 'commit' or type == 'new_commit':
            cm_id = row['commit_id']
            cm_text = row['summary'] + ' ' + row['diff']
            art = (cm_id, cm_text)
        elif type == 'link' or type == 'new_link':
            iss_id = row['issue_id']
            cm_id = row['commit_id']
            art = (iss_id, cm_id)
        else:
            raise Exception('wrong artifact type')
        arti.append(art)
    return arti

class Link(Dataset):

    def __init__(self, links):
        self.links = links

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        return self.links[idx]

class Link_with_idx(Dataset):

    def __init__(self, links):
        self.links = links

    def __len__(self):
        return len(self.links)

    def __getitem__(self, idx):
        return self.links[idx], idx

def exclude_and_sample(sample_pool, exclude, num):
    for id in exclude:
        sample_pool.remove(id)
    selected = random.sample(list(sample_pool), num)
    return selected

def clean_space(text):
    return " ".join(text.split())

class Examples:
    """
    Manage the examples read from raw dataset

    examples:
    valid_examples = CodeSearchNetReader(data_dir).get_examples(type="valid", num_limit=valid_num, summary_only=True)
    valid_examples = Examples(valid_examples)
    valid_examples.update_features(model)
    valid_examples.update_embd(model)

    """

    def __init__(self, raw_examples, max_seq_len=512):
        self.NL_index, self.PL_index, self.rel_index = self.__index_exmaple(raw_examples)
        self.max_seq_len = max_seq_len

    def __is_positive_case(self, nl_id, pl_id):
        if nl_id not in self.rel_index:
            return False
        rel_pls = set(self.rel_index[nl_id])
        return pl_id in rel_pls

    def __len__(self):
        return len(self.rel_index)

    def __index_exmaple(self, raw_examples):
        """
        Raw examples should be a dictionary with key "NL" for natural langauge and PL for programming language.
        Each {NL, PL} pair in same dictionary will be regarded as related ones and used as positive examples.
        :param raw_examples:
        :return:
        """
        rel_index = defaultdict(set)
        NL_index = dict()  # find instance by id
        PL_index = dict()

        # hanlde duplicated NL and PL with reversed index
        reverse_NL_index = dict()
        reverse_PL_index = dict()

        nl_id_max = 0
        pl_id_max = 0
        for r_exp in raw_examples:
            iss_id, iss_text = r_exp["NL"]
            nl_tks = clean_space(iss_text)
            cm_id, cm_text = r_exp["PL"]
            pl_tks = cm_text

            if nl_tks in reverse_NL_index:
                nl_id = reverse_NL_index[nl_tks]
            else:
                nl_id = nl_id_max
                nl_id_max += 1
                reverse_NL_index[nl_tks] = nl_id

            if pl_tks in reverse_PL_index:
                pl_id = reverse_PL_index[pl_tks]
            else:
                pl_id = pl_id_max
                pl_id_max += 1
                reverse_PL_index[pl_tks] = pl_id

            NL_index[nl_id] = {F_TOKEN: nl_tks, F_ID: nl_id, 'ori_id': iss_id}
            PL_index[pl_id] = {F_TOKEN: pl_tks, F_ID: pl_id, 'ori_id': cm_id}  # keep space for PL
            rel_index[nl_id].add(pl_id)
            nl_id += 1
            pl_id += 1
        return NL_index, PL_index, rel_index

    def _gen_feature(self, example, tokenizer):
        feature = tokenizer.encode_plus(example[F_TOKEN], max_length=self.max_seq_len, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False)
        res = {
            F_ID: example[F_ID],
            F_INPUT_ID: feature[F_INPUT_ID],
            F_ATTEN_MASK: feature[F_ATTEN_MASK]}
        return res

    def __update_feature_for_index(self, index, tokenizer):
        for v in tqdm(index.values(), desc="update feature"):
            f = self._gen_feature(v, tokenizer)
            id = f[F_ID]
            index[id][F_INPUT_ID] = f[F_INPUT_ID]
            index[id][F_ATTEN_MASK] = f[F_ATTEN_MASK]

    def update_features(self, model: BertClassifier):
        """
        Create or overwritten token_ids and attention_mask
        :param model:
        :return:
        """
        self.__update_feature_for_index(self.PL_index, model.get_pl_tokenizer())
        self.__update_feature_for_index(self.NL_index, model.get_nl_tokenizer())

    def __update_embd_for_index(self, index, sub_model):
        for id in tqdm(index, desc="update embedding"):
            feature = index[id]
            input_tensor = torch.tensor(feature[F_INPUT_ID]).view(1, -1).to(sub_model.device)
            mask_tensor = torch.tensor(feature[F_ATTEN_MASK]).view(1, -1).to(sub_model.device)
            embd = sub_model(input_tensor, mask_tensor)[0]
            embd_cpu = embd.to('cpu')
            index[id][F_EMBD] = embd_cpu

    def update_embd(self, model: BertClassifier):
        """
        Create or overwritten the embedding
        :param model:
        :return:
        """
        with torch.no_grad():
            model.eval()
            self.__update_embd_for_index(self.NL_index, model.get_nl_sub_model())
            self.__update_embd_for_index(self.PL_index, model.get_pl_sub_model())

    def get_retrivial_task_dataloader(self, batch_size):
        """create retrivial task"""
        res = []
        for nl_id in self.NL_index:
            for pl_id in self.PL_index:
                label = 1 if self.__is_positive_case(nl_id, pl_id) else 0
                res.append((nl_id, pl_id, label))
        dataset = DataLoader(Link(res), batch_size=batch_size)
        return dataset

    def get_chunked_retrivial_task_examples(self, chunk_query_num=-1, chunk_size=1000):
        """
        Cut the positive examples into chuncks. For EACH chunk generate queries at a size of query_num * chunk_size
        :param query_num: if query_num is -1 then create queries at a size of chunk_size * chunk_size
        :param chunk_size:
        :return:
        """
        rels = []
        for nid in self.rel_index:
            for pid in self.rel_index[nid]:
                rels.append((nid, pid))
        rel_dl = DataLoader(Link(rels), batch_size=chunk_size)
        examples = []
        for batch in rel_dl:
            batch_query_idx = 0
            nids, pids = batch[0].tolist(), batch[1].tolist()
            for nid in nids:
                batch_query_idx += 1
                if chunk_query_num != -1 and batch_query_idx > chunk_query_num:
                    break
                for pid in pids:
                    label = 1 if self.__is_positive_case(nid, pid) else 0
                    examples.append((nid, pid, label))
        return examples

    def id_pair_to_embd_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_tensor = self._id_to_embd(nl_id_tensor, self.NL_index)
        pl_tensor = self._id_to_embd(pl_id_tensor, self.PL_index)
        return nl_tensor, pl_tensor

    def id_pair_to_feature_pair(self, nl_id_tensor: Tensor, pl_id_tensor: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert id pairs into embdding pairs"""
        nl_input_tensor, nl_att_tensor = self._id_to_feature(nl_id_tensor, self.NL_index)
        pl_input_tensor, pl_att_tensor = self._id_to_feature(pl_id_tensor, self.PL_index)
        return nl_input_tensor, nl_att_tensor, pl_input_tensor, pl_att_tensor

    def _id_to_feature(self, id_tensor: Tensor, index):
        input_ids, att_masks = [], []
        for id in id_tensor.tolist():
            input_ids.append(torch.tensor(index[id][F_INPUT_ID]))
            if F_ATTEN_MASK in index[id]:
                att_masks.append(torch.tensor(index[id][F_ATTEN_MASK]))
        input_tensor = torch.stack(input_ids)
        att_tensor = None
        if att_masks:
            att_tensor = torch.stack(att_masks)
        return input_tensor, att_tensor

    def _id_to_embd(self, id_tensor: Tensor, index):
        embds = []
        for id in id_tensor.tolist():
            embds.append(index[id][F_EMBD])
        embd_tensor = torch.stack(embds)
        return embd_tensor

    def random_neg_sampling_dataloader(self, batch_size, data_loader_flag=True):
        pos, neg = [], []
        for nl_id in self.rel_index:
            pos_pl_ids = self.rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((nl_id, n_id, 0))
        if data_loader_flag:
            links = Link(pos + neg)
            sampler = RandomSampler(links)
            dataset = DataLoader(links, batch_size=batch_size, sampler=sampler)
            return dataset
        else:
            return pos + neg

    def update_plabels(self, p_labels_data, p_cutoff):
        pseudo_rel_index = defaultdict(set)
        for nl_id, pl_id, sim_score, p_label in p_labels_data:
            if p_label == 1 and sim_score[1] > p_cutoff:
                pseudo_rel_index[nl_id].add(pl_id)
        self.pseudo_rel_index = pseudo_rel_index

    def random_neg_sampling_dataloader_by_plabels(self, batch_size):
        pos, neg = [], []
        for nl_id in self.pseudo_rel_index:
            pos_pl_ids = self.pseudo_rel_index[nl_id]
            for p_id in pos_pl_ids:
                pos.append((nl_id, p_id, 1))
            sample_num = len(pos_pl_ids)
            sel_neg_ids = exclude_and_sample(set(self.PL_index.keys()), pos_pl_ids, sample_num)
            for n_id in sel_neg_ids:
                neg.append((nl_id, n_id, 0))
        links = Link(pos + neg)
        sampler = RandomSampler(links)
        dataset = DataLoader(links, batch_size=batch_size, sampler=sampler)
        return dataset

