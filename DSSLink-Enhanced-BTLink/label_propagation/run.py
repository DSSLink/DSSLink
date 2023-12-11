import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import datetime
from train_stage1 import main as train1
from train_deepssl import main as train2

current_dir = os.path.dirname(os.path.abspath(__file__))

datasets_dir = 'Your datasets_dir'
output_dir = 'Your output_dir'
result_dir = 'Your result_dir'
datasets = [
    {'pro': 'Avro', 'key': 'AVRO'},
    {'pro': 'Beam', 'key': 'BEAM'},
    {'pro': 'Buildr', 'key': 'BUILDR'},
    {'pro': 'Giraph', 'key': 'GIRAPH'},
    {'pro': 'ant-ivy', 'key': 'IVY'},
    {'pro': 'Isis', 'key': 'ISIS'},
    {'pro': 'logging-log4net', 'key': 'LOG4NET'},
    {'pro': 'Nutch', 'key': 'NUTCH'},
    {'pro': 'OODT', 'key': 'OODT'},
    {'pro': 'Tez', 'key': 'TEZ'},
    {'pro': 'Tika', 'key': 'TIKA'},
]


def main():
    for dataset in datasets:
        pro = dataset['pro']
        key = dataset['key']
        exec(pro, key, 'base')  # BTLink Base Model with 20 epochs
        exec(pro, key, 'pre')  # BTLink Pre-trained Model with 10 epochs
        exec(pro, key, 'deepssl')  # BTLink DeepSSL Model with 10 epochs


def exec(pro, key, task_type):
    desc = f'pro: {pro}, key: {key}, task_type: {task_type}'
    if task_type == 'base' or task_type == 'pre':
        log_write(f'Start training the {task_type} model: {desc}')
        model_name = train1(datasets_dir, output_dir, result_dir, pro, key, task_type)
        sys_argv_clear()
        log_write(f'The {task_type} model is trained: {desc}')
    else:
        log_write(f'Start training the {task_type} model: {desc}')
        train2(datasets_dir, output_dir, result_dir, pro, key, task_type)
        sys_argv_clear()
        log_write(f'The {task_type} model is trained: {desc}')


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