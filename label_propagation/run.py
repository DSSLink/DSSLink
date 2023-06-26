import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import shutil
import datetime
from train_stage1 import main as train1
from train_deepssl import main as train2
from eval_stage import main as eval

current_dir = os.path.dirname(os.path.abspath(__file__))

runs_path = os.path.join(current_dir, 'runs')
eval_path = os.path.join(current_dir, 'evaluation')

datasets_dir = '..'
outputs_dir = '..'


def main():
    dname = 'flask'
    epochs = 100
    for lab_frac in ['_num10', 10, 30, 50, 70]:
        for group in range(1, 6):
            exec(dname, str(group), lab_frac, stage='ori', epochs=epochs)  # T-BERT 100 epochs (DSSLink pre-training)
            exec(dname, str(group), lab_frac, stage='ssl', epochs=epochs)   # DSSLink 100 epochs (Deep Semi-Supervised Learning)
            exec(dname, str(group), lab_frac, stage='ori', epochs=epochs * 2)  # T-BERT 200 epochs (Original T-BERT)

    # Reduce the amount of unlabeled data to train DSSLink
    for lab_frac in ['_num10', 10, 30, 50, 70]:
        for group in range(1, 6):
            exec(dname, str(group), lab_frac, stage='ssl', epochs=epochs, task_type='partial')

    # Randomly select the unlabeled software artifact pairs to train DSSLink
    for lab_frac in ['_num10', 10, 30, 50, 70]:
        for group in range(1, 6):
            exec(dname, str(group), lab_frac, stage='ssl', epochs=epochs, task_type='random')

def exec(dname, group, lab_frac, stage, epochs, task_type=None):
    desc = f'dataset: {dname}, group: {group}, labfrac: {lab_frac}, epochs: {epochs}'
    if stage == 'ori':
        log_write(f'Start training the original model: {desc}')
        lab_dtype = str(lab_frac)
        model_name = train1(datasets_dir, outputs_dir, dname, group, lab_dtype, epochs)
        sys_argv_clear()
        # Delete the runs and eval folders
        shutil.rmtree(runs_path)
        shutil.rmtree(eval_path)
        eval(datasets_dir, outputs_dir, dname, group, model_name)
        sys_argv_clear()
        log_write(f'The original model is trained: {desc}')
    else:
        assert stage == 'ssl'
        log_write(f'Start training the DSSLink model: {desc}')
        lab_dtype = str(lab_frac)
        unlab_dtype = str(100 - lab_frac) if isinstance(lab_frac, int) else str(lab_frac)
        model_name = train2(datasets_dir, outputs_dir, dname, group, lab_dtype, unlab_dtype, epochs=epochs, task_type=task_type)
        sys_argv_clear()
        # Delete the runs and eval folders
        shutil.rmtree(runs_path)
        shutil.rmtree(eval_path)
        eval(datasets_dir, outputs_dir, dname, group, model_name)
        sys_argv_clear()
        log_write(f'The DSSLink model is trained: {desc}')

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