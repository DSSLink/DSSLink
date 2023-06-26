import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import shutil
import datetime
from Prediction_by_VSM import main as predicting
from unlabeled_data_labeling import main as labeling


current_dir = os.path.dirname(os.path.abspath(__file__))

runs_path = os.path.join(current_dir, 'runs')
eval_path = os.path.join(current_dir, 'evaluation')

datasets_dir = '..'


def main():
    dname = 'flask'
    frac = 1.1
    for lab_frac in ['_num10', 10, 30, 50, 70]:
        for group in range(1, 6):
            exec(dname, str(group), lab_frac, frac)

def exec(dname, group, lab_frac, frac):
    desc = f'dataset: {dname}, group: {group}, labfrac: {lab_frac}'
    log_write(f'tracefun start: {desc}')
    lab_dtype = str(lab_frac)
    unlab_dtype = str(100 - lab_frac) if isinstance(lab_frac, int) else str(lab_frac)
    lab_links_num = predicting(datasets_dir, dname, group, lab_dtype, unlab_dtype, use_new_links=True)
    need_add_link_num = int(lab_links_num * frac)
    labeling(datasets_dir, dname, group, lab_dtype, need_add_link_num=need_add_link_num)
    log_write(f'tracefun finish: {desc}')

def log_write(text):
    log_file = 'log.txt'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf8') as f:
        f.write('%s %s\n' % (current_time, text))
    print(text)

def sys_argv_clear():
    for i in sys.argv[:]:
        if i.startswith('--'):
            sys.argv.remove(i)

if __name__ == '__main__':
    main()