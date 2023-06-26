import os
import random
import pandas as pd
import numpy as np
import collections

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel("INFO")


def main(datasets_dir, dname, group, lab_dtype, need_add_link_num):
    dataset_dir = f'{datasets_dir}/{dname}/{group}'
    labeling = Labeling(dataset_dir, lab_dtype)
    labeling.run(need_add_link_num=need_add_link_num)
    labeling.add_data2train_data()

class Labeling:

    def __init__(self, dataset_dir, lab_dtype):
        self.dataset_dir = dataset_dir
        self.lab_dataset_dir = os.path.join(dataset_dir, f'train_lab{lab_dtype}')
        self.similarity_list_dir = os.path.join(dataset_dir, f'train_lab{lab_dtype}_tracefun')

    def run(self, need_add_link_num):
        self.load_data()
        new_links = self.labeling_by_num(need_add_link_num)
        csv_path = os.path.join(self.similarity_list_dir, 'labeling_result.csv')
        if os.path.exists(csv_path):
            os.remove(csv_path)
        pd.DataFrame(new_links, columns=['issue_id', 'commit_id']).to_csv(csv_path, index=False)

    def add_data2train_data(self):
        new_links_df = pd.read_csv(os.path.join(self.similarity_list_dir, 'labeling_result.csv'))
        df = pd.concat([self.train_lab_df, new_links_df])
        df.to_csv(os.path.join(self.similarity_list_dir, 'link.csv'), index=False)

    def load_data(self):
        self.iss_res_df = pd.read_csv(os.path.join(self.similarity_list_dir, 'issue_similarity_list.csv'))
        self.com_res_df = pd.read_csv(os.path.join(self.similarity_list_dir, 'commit_similarity_list.csv'))
        self.train_lab_df = pd.read_csv(os.path.join(self.lab_dataset_dir, 'link.csv'))
        self.s_to_t = collections.defaultdict(set)  # Target artifact group for each source artifact marked
        self.t_to_s = collections.defaultdict(set)  # The source artifact group for each target artifact marked
        for s_id, t_id in self.train_lab_df[['issue_id', 'commit_id']].values:
            self.s_to_t[s_id].add(t_id)
            self.t_to_s[t_id].add(s_id)
        self.result_str = [f'{s_id}_{t_id}' for s_id, t_id in self.train_lab_df[['issue_id', 'commit_id']].values]

    def labeling_by_threshold(self, threshold):
        source_sim_df = self.iss_res_df.copy()
        target_sim_df = self.com_res_df.copy()
        source_sim_df = source_sim_df.loc[source_sim_df['pred'] > threshold]
        target_sim_df = target_sim_df.loc[target_sim_df['pred'] > threshold]
        source_sim_df.sort_values('pred', ascending=False, inplace=True)
        target_sim_df.sort_values('pred', ascending=False, inplace=True)

        link_from_source_data = list()  # Traceability links generated from similar source artifacts
        link_from_target_data = list()  # Traceability links generated from similar target artifacts

        for lab_id, unlab_id, pred in source_sim_df.values.tolist():
            links = self.create_link_from_source(lab_id, unlab_id)
            link_from_source_data.extend(links)

        for lab_id, unlab_id, pred in target_sim_df.values.tolist():
            links = self.create_link_from_target(lab_id, unlab_id)
            link_from_target_data.extend(links)

        # count
        link_from_source_num = len(link_from_source_data)
        link_from_target_num = len(link_from_target_data)
        logger.info(f'Added number of links via source artifact similarity:{link_from_source_num}')
        logger.info(f'Added number of links by target artifact similarity:{link_from_target_num}')
        # take union
        new_links = list(set(link_from_source_data).union(set(link_from_target_data)))
        # deduplication
        new_links = list({}.fromkeys(new_links).keys())
        # Convert from id string to id list
        new_links = [id_str.split('_') for id_str in new_links]
        logger.info(f'Number of new labeling links：{len(new_links)}')
        return new_links

    def labeling_by_num(self, need_add_link_num):
        source_sim_df = self.iss_res_df.copy()
        target_sim_df = self.com_res_df.copy()
        source_sim_df.sort_values('pred', ascending=False, inplace=True)
        target_sim_df.sort_values('pred', ascending=False, inplace=True)

        '**********Generate traceability links from unlabeled data**********'
        logger.info('Generate traceability links from unlabeled data')
        link_from_source_data = list()  # Traceability links generated from similar source artifacts
        link_from_target_data = list()  # Traceability links generated from similar target artifacts
        similar_source_index, similar_target_index = 0, 0  # Index of similar pairs of source and target artifacts
        source_sim_pred, target_sim_pred = 1, 1  # Minimum similarity of source and target artifacts
        while need_add_link_num > 0:  # Exit the loop when the number of new links to be added is less than or equal to 0
            similar_source = source_sim_df.iloc[similar_source_index]  # Similar source artifacts
            similar_target = target_sim_df.iloc[similar_target_index]  # Similar target artifacts
            source_pred = float(similar_source['pred'])  # Similarity of similar source artifacts
            target_pred = float(similar_target['pred'])  # Similarity of similar target artifacts
            if source_pred > target_pred:  # Use source artifact kick to generate traceability links when similar source artifacts are more similar
                links = self.create_link_from_source(similar_source['lab_id'], similar_source['unlab_id'])
                link_from_source_data.extend(links)
                similar_source_index += 1  # Source Artifact Similar Pair Index+1
                need_add_link_num -= len(links)  # The number of new links that need to be added minus the number of new links added in this loop
                source_sim_pred = source_pred
            elif source_pred < target_pred:  # Use source artifact kick to generate traceability links when similar target artifacts are more similar
                links = self.create_link_from_target(similar_target['lab_id'], similar_target['unlab_id'])
                link_from_target_data.extend(links)
                similar_target_index += 1  # Target Artifact Artifact Similar Pair Index+1
                need_add_link_num -= len(links)  # The number of new links that need to be added minus the number of new links added in this loop
                target_sim_pred = target_pred
            else:
                assert (source_pred == target_pred)  # The similarity between the two workpieces is equal
                ls_f_s = self.create_link_from_source(similar_source['lab_id'], similar_source['unlab_id'])
                ls_f_t = self.create_link_from_target(similar_target['lab_id'], similar_target['unlab_id'])
                link_from_source_data.extend(ls_f_s)
                link_from_target_data.extend(ls_f_t)
                similar_source_index += 1  # Source Artifact Similar Pair Index+1
                similar_target_index += 1  # Target Artifact Artifact Similar Pair Index+1
                need_add_link_num = need_add_link_num - len(ls_f_s) - len(ls_f_t)  # The number of new links that need to be added minus the number of new links added in this loop
                source_sim_pred, target_sim_pred = source_pred, target_pred
        logger.info(f'The minimum source artifact similarity used for identification is: {source_sim_pred}')
        logger.info(f'The minimum target artifact similarity used for identification is: {target_sim_pred}')
        # deduplication
        link_from_source_data = set(link_from_source_data)
        link_from_target_data = set(link_from_target_data)
        # count
        link_from_source_num = len(link_from_source_data)
        link_from_target_num = len(link_from_target_data)
        logger.info(f'Added number of links via source artifact similarity:{link_from_source_num}')
        logger.info(f'Added number of links by target artifact similarity:{link_from_target_num}')
        # take union
        new_links = list(link_from_source_data.union(link_from_target_data))
        # deduplication
        new_links = list({}.fromkeys(new_links).keys())
        # Convert from id string to id list
        new_links = [id_str.split('_') for id_str in new_links]
        logger.info(f'Number of new labeling links：{len(new_links)}')
        return new_links

    def create_link_from_source(self, lab_id, unlab_id):
        links = list()
        for t_id in self.s_to_t.get(lab_id, []):
            id_str = f'{unlab_id}_{t_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        return links

    def create_link_from_target(self, lab_id, unlab_id):
        links = list()
        for s_id in self.t_to_s.get(lab_id, []):
            id_str = f'{s_id}_{unlab_id}'
            if id_str not in self.result_str:
                links.append(id_str)
        return links


if __name__ == '__main__':
    main('flask', '1', 20)


