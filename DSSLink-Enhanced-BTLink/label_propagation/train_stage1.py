from __future__ import absolute_import, division, print_function
import sys
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,\
    SequentialSampler, RandomSampler
from tqdm import tqdm
from model import BTModel
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, AutoModel)
from utils import getargs, set_seed, TextDataset, Artifacts, ArtifactsDataset
import warnings
warnings.filterwarnings(action='ignore')


def train(args, train_dataset, model, tokenizer, task_type):
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'], dtype=object)
    torch.set_grad_enabled(True)
    ''' Train the model '''
    text_examples, code_examples = train_dataset.generate_features(train_dataset.pair_by_DRNS())
    pairs_by_DRNS = ArtifactsDataset(text_examples, code_examples)
    train_sampler = RandomSampler(pairs_by_DRNS)
    train_dataloader = DataLoader(pairs_by_DRNS, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=1, pin_memory=True)
    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.pro + '_DEV.csv'))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    print('********** Running training **********')
    print('  Num examples = {}'.format(len(pairs_by_DRNS)))
    print('  Num Epochs = {}'.format(args.num_train_epochs))
    print('  batch size = {}'.format(args.train_batch_size))
    print('  Total optimization steps = {}'.format(max_steps))
    best_f = 0.0
    model.zero_grad()
    model.train()
    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            text_inputs = batch[0].to(args.device)
            code_inputs = batch[1].to(args.device)
            labels = batch[2].to(args.device)

            loss, logits = model(text_inputs, code_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description('epoch {} loss {}'.format(idx, round(float(np.mean(losses)), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, eval_dataset)
        for key, value in results.items():
            print('-'*10 + '  {} = {}'.format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + '  {} = {}'.format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_f1'] >= best_f:
            best_f = results['eval_f1']
            print('  ' + '*' * 20)
            print('  Best f1: {}'.format(round(best_f, 4)))
            print('  ' + '*' * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format(f'model_{task_type}.bin'))
            torch.save(model, output_dir)
            print('Saving model checkpoint to {}'.format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        result_dir = os.path.join(args.result_dir, args.pro)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        dfScores.to_csv(os.path.join(result_dir, args.pro + f'Epoch_Metrics_{task_type}.csv'), index=False)


def evaluate(args, model, eval_dataset):
    stime = datetime.datetime.now()
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    args.seed += 3
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                 pin_memory=True)

    # Eval!
    print('***** Running evaluation *****')
    print('  Num examples = {}'.format(len(eval_dataset)))
    print('  Batch size = {}'.format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        text_inputs = batch[0].to(args.device)
        code_inputs = batch[1].to(args.device)
        label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, code_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    print('Predictions', preds[:25])
    print('Labels:', labels[:25])
    etime = datetime.datetime.now()
    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        'eval_loss': float(perplexity),
        'eval_time': float(eval_time),
        'eval_acc': round(float(eval_acc), 4),
        'eval_precision': round(eval_precision, 4),
        'eval_recall': round(eval_recall, 4),
        'eval_f1': round(eval_f1, 4),
        'eval_auc': round(eval_auc, 4),
        'eval_mcc': round(eval_mcc, 4),
        'eval_brier': round(eval_brier, 4),
        'eval_pf': round(eval_pf, 4),
    }
    return result


def test(args, model, tokenizer, stime, task_type):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.pro + '_TEST.csv'))
    args.seed += 3
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print('********** Running Test **********')
    print('  Num examples = {}'.format(len(eval_dataset)))
    print('  Batch size = {}'.format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    issues = []
    commits = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        text_inputs = batch[0].to(args.device)
        code_inputs = batch[1].to(args.device)
        label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(text_inputs, code_inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            for iss, com in zip(batch[3][0], batch[4][0]):
                issues.append(iss)
                commits.append(com)
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp/(fp+tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        'eval_loss': float(perplexity),
        'eval_time': float(eval_time),
        'eval_acc': round(float(eval_acc), 4),
        'eval_precision': round(eval_precision, 4),
        'eval_recall': round(eval_recall, 4),
        'eval_f1': round(eval_f1, 4),
        'eval_auc': round(eval_auc, 4),
        'eval_mcc': round(eval_mcc, 4),
        'eval_brier': round(eval_brier, 4),
        'eval_pf': round(eval_pf, 4),
    }
    print(preds[:25], labels[:25])
    print('********** Test results **********')
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'], dtype=object)
    for key in sorted(result.keys()):
        print('-'*10 + '  {} = {}'.format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]
    result_dir = os.path.join(args.result_dir, args.pro)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    dfScores.to_csv(os.path.join(result_dir, args.pro + f'_Metrics_{task_type}.csv'), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'
    logits4class0, logits4class1 = \
        [logits[iclass][0] for iclass in range(len(logits))],\
        [logits[iclass][1] for iclass in range(len(logits))]
    df = pd.DataFrame(np.transpose([issues, commits, logits4class0, logits4class1, preds, labels]),
                      columns=['Issue_Key', 'Commit_SHA', '0_logit', '1_logit', 'preds', 'labels'])
    df.to_csv(os.path.join(result_dir, args.pro + f'_Prediction_{task_type}.csv'), index=False)


def main(datasets_dir, outputs_dir, result_dir, pro, key, task_type):
    epochs = 20 if task_type == 'base' else 10
    sys.argv.append(f'--data_dir={datasets_dir}')
    sys.argv.append(f'--output_dir={outputs_dir}')
    sys.argv.append(f'--result_dir={result_dir}')
    sys.argv.append(f'--pro={pro}')
    sys.argv.append(f'--key={key}')

    sys.argv.append('--text_model_path=roberta-large')
    sys.argv.append('--code_model_path=codeBERT')
    sys.argv.append('--tokenizer_name=roberta-large')

    sys.argv.append('--max_seq_length=512')
    sys.argv.append(f'--num_train_epochs={epochs}')
    sys.argv.append('--train_batch_size=4')
    sys.argv.append('--eval_batch_size=4')
    # sys.argv.append('--learning_rate=1r-5')
    # sys.argv.append('--weight_decay=0.0')
    # sys.argv.append('--seed=42')


    print('======BTLink BEGIN...======' * 20)
    stime = datetime.datetime.now()
    args = getargs()
    args.lab_frac = 0.1
    print(args.key)
    print(f'===Project:{args.pro}---{args.pro}==='*5)
    print('device: {}, n_gpu: {}'.format(args.device, args.n_gpu) )
    # Set seed
    set_seed(args.seed)

    # Roberta
    config = RobertaConfig.from_pretrained(args.text_model_path)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    textEncoder = AutoModel.from_pretrained(args.text_model_path, config=config)

    # CodeBERT
    config4Code = RobertaConfig.from_pretrained(args.code_model_path)
    config4Code.num_labels = 2
    codeEncoder = AutoModel.from_pretrained(args.code_model_path, config=config4Code)
    model = BTModel(textEncoder, codeEncoder, config.hidden_size, config4Code.hidden_size, args.num_class)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    print('Training/evaluation parameters {}'.format(args))
    # Training
    if args.do_train:
        # sample - dataset
        file_path = os.path.join(args.data_dir, args.pro + '_TRAIN.csv')
        SplitLabAndUnlab(args, file_path, lab_frac=args.lab_frac)
        train_dataset = Artifacts(tokenizer, args, os.path.join(args.data_dir, args.pro + f'_TRAIN_lab{int(args.lab_frac * 100)}.csv'))
        args.seed += 3
        train(args, train_dataset, model, tokenizer, task_type)

    if args.do_test:
        checkpoint_prefix = args.pro + f'_checkpoint-best/model_{task_type}.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        test(args, model, tokenizer, stime, task_type)


def SplitLabAndUnlab(args, train_file_path, lab_frac):
    train_df = pd.read_csv(train_file_path)
    label1_df = train_df.loc[train_df['label'] == 1]
    label0_df = train_df.loc[train_df['label'] == 0]
    small_label1_df = label1_df.sample(frac=lab_frac)
    small_label0_df = label0_df.sample(frac=lab_frac)
    lab_df = pd.concat([small_label1_df, small_label0_df], axis=0)
    unlab_df = train_df[~train_df.index.isin(lab_df.index)]

    lab_path = os.path.join(args.data_dir, args.pro + f'_TRAIN_lab{int(lab_frac * 100)}.csv')
    unlab_path = os.path.join(args.data_dir, args.pro + f'_TRAIN_unlab{int((1 - lab_frac) * 100)}.csv')
    if not os.path.exists(lab_path):
        lab_df.to_csv(lab_path, index=False)
    if not os.path.exists(unlab_path):
        unlab_df.to_csv(unlab_path, index=False)


if __name__ == '__main__':
    main()
