import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import shutil
import datetime
from train_stage import main as train
from eval_stage import main as eval

current_dir = os.path.dirname(os.path.abspath(__file__))

runs_path = os.path.join(current_dir, 'runs')
eval_path = os.path.join(current_dir, 'evaluation')

datasets_dir = 'Your datasets_dir'
outputs_dir = 'Your outputs_dir'
datasets = ['flask', 'pgcli', 'keras', 'scrapy']


def main():
    for dname in datasets:
        epochs = 200
        for lab_frac in ['_num10', 10, 30, 50, 70]:
            for group in range(1, 6):
                exec(dname, str(group), lab_frac, epochs=epochs)

def exec(dname, group, lab_frac, epochs):
    desc = f'dataset: {dname}, group: {group}, labfrac: {lab_frac}, epochs: {epochs}'
    log_write(f'Start training the TraceFUN model: {desc}')
    model_name = train(datasets_dir, outputs_dir, dname, group, f'{lab_frac}_tracefun', epochs=epochs, use_new_links=True)
    sys_argv_clear()
    # Delete the runs and eval folders
    shutil.rmtree(runs_path)
    shutil.rmtree(eval_path)
    eval(datasets_dir, outputs_dir, dname, group, model_name)
    sys_argv_clear()
    log_write(f'The TraceFUN model is trained: {desc}')

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