import os
import shutil
import random
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim import corpora, models, matutils
from TextPreprocess import preprocessor


def main(datasets_dir, dname, group, lab_dtype, unlab_dtype, use_new_links):
    dataset_dir = f'{datasets_dir}/{dname}/{group}'
    set_seed(42)
    predict = Prediction(dataset_dir, lab_dtype, unlab_dtype)
    lab_links_num = predict.run(use_new_links)
    return lab_links_num

def set_seed(seed):
    random.seed(seed)  # set random seed for python
    np.random.seed(seed)  # set random seed for numpy

class Prediction:

    def __init__(self, dataset_dir, lab_dtype, unlab_dtype):
        self.dataset_dir = dataset_dir
        self.lab_dataset_dir = os.path.join(dataset_dir, f'train_lab{lab_dtype}')
        self.unlab_dataset_dir = os.path.join(dataset_dir, f'train_unlab{unlab_dtype}')
        self.output_dir = os.path.join(dataset_dir, f'train_lab{lab_dtype}_tracefun')

    def run(self, use_new_links):
        s_artifacts, t_artifacts, lab_links_num = self.load_artifacts(use_new_links=use_new_links)
        documents = [doc for doc in s_artifacts.values()] + [doc for doc in t_artifacts.values()]
        self.build_VSM(documents)
        print('build vsm ok')

        iss_pairs, com_pairs = self.read_links(s_artifacts, t_artifacts)
        iss_res_df = self.predict(iss_pairs, s_artifacts)
        com_res_df = self.predict(com_pairs, t_artifacts)

        shutil.rmtree(self.output_dir, ignore_errors=True)
        os.mkdir(self.output_dir)

        iss_res_df.to_csv(os.path.join(self.output_dir, 'issue_similarity_list.csv'), index=False)
        com_res_df.to_csv(os.path.join(self.output_dir, 'commit_similarity_list.csv'), index=False)
        print(f'predict finish, {self.output_dir}, lab_links_num: {lab_links_num}')
        return lab_links_num

    def load_artifacts(self, use_new_links):
        s_artifacts = dict()
        t_artifacts = dict()
        issues = self.read_artifacts(os.path.dirname(self.dataset_dir), 'issue')
        commits = self.read_artifacts(os.path.dirname(self.dataset_dir), 'commit')
        lab_links = self.read_artifacts(self.lab_dataset_dir, 'link')
        unlab_links = self.read_artifacts(self.unlab_dataset_dir, 'link')
        issue_index = {iss_id: iss_text for iss_id, iss_text in issues}
        commit_index = {cm_id: cm_text for cm_id, cm_text in commits}
        for iss_id, cm_id in lab_links:
            s_artifacts[iss_id] = preprocessor.run(issue_index[iss_id]).split()
            t_artifacts[cm_id] = preprocessor.run(commit_index[cm_id]).split()
        for iss_id, cm_id in unlab_links:
            s_artifacts[iss_id] = preprocessor.run(issue_index[iss_id]).split()
            t_artifacts[cm_id] = preprocessor.run(commit_index[cm_id]).split()
        if use_new_links:
            print('Use new links')
            new_issues = self.read_artifacts(os.path.dirname(self.dataset_dir), 'new_issue')
            new_commits = self.read_artifacts(os.path.dirname(self.dataset_dir), 'new_commit')
            for iss_id, iss_text in new_issues:
                s_artifacts[iss_id] = preprocessor.run(iss_text).split()
            for cm_id, cm_text in new_commits:
                t_artifacts[cm_id] = preprocessor.run(cm_text).split()
        return s_artifacts, t_artifacts, len(lab_links)

    def read_artifacts(self, dataset_dir, type):
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

    def read_links(self, s_artifacts, t_artifacts):
        train_lab_df = pd.read_csv(os.path.join(self.lab_dataset_dir, 'link.csv'))
        lab_iss_ids = list({}.fromkeys(train_lab_df.issue_id.to_list()).keys())
        lab_com_ids = list({}.fromkeys(train_lab_df.commit_id.to_list()).keys())
        iss_ids = list(s_artifacts.keys())
        com_ids = list(t_artifacts.keys())
        unlab_iss_ids = list(set(iss_ids).difference(lab_iss_ids))
        unlab_com_ids = list(set(com_ids).difference(lab_com_ids))
        iss_pairs = self.gen_arti_pairs(lab_iss_ids, unlab_iss_ids)
        com_pairs = self.gen_arti_pairs(lab_com_ids, unlab_com_ids)

        return iss_pairs, com_pairs

    def gen_arti_pairs(self, lab_ids, unlab_ids):
        arti_pairs = list()
        for lab_id, unlab_id in itertools.product(lab_ids, unlab_ids):
            arti_pairs.append([lab_id, unlab_id])
        return arti_pairs

    def predict(self, arti_pairs, arti_index):
        results = list()
        for index in tqdm(range(len(arti_pairs)), '预测数据'):
            lab_id, unlab_id = arti_pairs[index]
            pred = self.similarity(arti_index[lab_id], arti_index[unlab_id])
            results.append([lab_id, unlab_id, pred])
        res_df = pd.DataFrame(results, columns=['lab_id', 'unlab_id', 'pred'])
        return res_df

    def build_VSM(self, documents):
        dictionary = corpora.Dictionary(documents)  # generate dict
        corpus = [dictionary.doc2bow(doc) for doc in documents]  # generate corpus
        tfidf_model = models.TfidfModel(corpus, id2word=dictionary)  # build vsm
        self.dictionary = dictionary
        self.vsm = tfidf_model

    def similarity(self, source, target):
        source_bow = self.dictionary.doc2bow(source)
        target_bow = self.dictionary.doc2bow(target)
        source_vec = self.vsm[source_bow]
        target_vec = self.vsm[target_bow]
        return matutils.cossim(source_vec, target_vec)


if __name__ == '__main__':
    main('flask', '1', 'lab10', 'unlab90')