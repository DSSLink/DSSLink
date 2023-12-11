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


def train(args, train_lab_dataset, train_unlab_dataset, model, tokenizer):
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'], dtype=object)
    torch.set_grad_enabled(True)
    """ Train the model """
    # 根据DRNS生成的已标记软件制品对
    text_examples, code_examples = train_lab_dataset.generate_features(train_lab_dataset.pair_by_DRNS())
    lab_artifacts = ArtifactsDataset(text_examples, code_examples)
    train_lab_sampler = RandomSampler(lab_artifacts)
    train_lab_dataloader = DataLoader(lab_artifacts, sampler=train_lab_sampler, batch_size=args.train_lab_batch_size, num_workers=1, pin_memory=True)

    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.pro + '_DEV.csv'))

    # 根据日期规则生成的未标记软件制品对
    text_examples, code_examples = train_unlab_dataset.generate_features(train_unlab_dataset.pair_by_datetime())
    unlab_artifacts_by_datetime = ArtifactsDataset(text_examples, code_examples)
    unlab_dataloader_by_datetime = DataLoader(unlab_artifacts_by_datetime, batch_size=args.train_batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_lab_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    # Train!
    print("********** Running training **********")
    print("  Num examples = {}".format(len(lab_artifacts)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  batch size = {}".format(args.train_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    best_f = 0.0
    model.zero_grad()
    model.train()
    for idx in range(args.num_train_epochs):
        # update_plabels
        plabels_data = extract_pseudo_labels(unlab_dataloader_by_datetime, model, args)
        pos_plabel_num = train_unlab_dataset.update_plabels(plabels_data, args.p_cutoff)
        if pos_plabel_num:
            text_examples, code_examples = train_unlab_dataset.generate_unlab_features(train_unlab_dataset.pair_by_DRNS_and_plabels(), len(train_lab_dataloader)*args.train_unlab_batch_size)
            unlab_artifacts_by_plabels = ArtifactsDataset(text_examples, code_examples)
            unlab_sampler_by_plabels = RandomSampler(unlab_artifacts_by_plabels)
            unlab_dataloader_by_plabels = DataLoader(unlab_artifacts_by_plabels, batch_size=args.train_unlab_batch_size, sampler=unlab_sampler_by_plabels)
            unlab_iter = iter(unlab_dataloader_by_plabels)

        bar = tqdm(train_lab_dataloader, total=len(train_lab_dataloader))
        losses = []
        for step, lab_batch in enumerate(bar):
            lab_text_inputs = lab_batch[0].to(args.device)
            lab_code_inputs = lab_batch[1].to(args.device)
            lab_labels = lab_batch[2].to(args.device)
            if pos_plabel_num:
                num_lb = lab_labels.size(0)

                try:
                    unlab_batch = next(unlab_iter)
                except Exception as e:
                    unlab_iter = iter(unlab_dataloader_by_plabels)
                    unlab_batch = next(unlab_iter)

                unlab_text_inputs = unlab_batch[0].to(args.device)
                unlab_code_inputs = unlab_batch[1].to(args.device)
                unlab_labels = unlab_batch[2].to(args.device)

                text_inputs = torch.cat([lab_text_inputs, unlab_text_inputs], dim=0)
                code_inputs = torch.cat([lab_code_inputs, unlab_code_inputs], dim=0)

                logits = model.get_logits(text_inputs, code_inputs)
                logits_lb = logits[:num_lb]
                logits_ulb = logits[num_lb:]

                sup_loss = criterion(logits_lb.view(-1, 2), lab_labels.view(-1))
                unsup_loss = criterion(logits_ulb.view(-1, 2), unlab_labels.view(-1))
                lambda_u = 1.0
                loss = sup_loss + lambda_u * unsup_loss
            else:
                loss, logits = model(lab_text_inputs, lab_code_inputs, lab_labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(float(np.mean(losses)), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        results = evaluate(args, model, eval_dataset)
        for key, value in results.items():
            print('-'*10 + "  {} = {}".format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + "  {} = {}".format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_f1'] >= best_f:
            best_f = results['eval_f1']
            print("  " + "*" * 20)
            print("  Best f1: {}".format(round(best_f, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model_st.bin'))
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        result_dir = os.path.join(args.result_dir, args.pro)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        dfScores.to_csv(os.path.join(result_dir, args.pro + "Epoch_Metrics_st.csv"), index=False)


def extract_pseudo_labels(train_unlab_dataloader, model, args):
    model.eval()
    plabels_data = []
    for step, unlab_batch in enumerate(tqdm(train_unlab_dataloader, total=len(train_unlab_dataloader), desc='Extracting Pseudo Labels')):
        text_inputs = unlab_batch[0].to(args.device)
        code_inputs = unlab_batch[1].to(args.device)
        labels = unlab_batch[2]
        iss_ids = unlab_batch[3]
        comm_ids = unlab_batch[4]
        with torch.no_grad():
            logits = model.get_logits(text_inputs, code_inputs)
            sim_scores = torch.softmax(logits, dim=-1)
            plabels = torch.max(sim_scores, dim=-1)[1]
        for iss_id, comm_id, sim_score, plabel, label in zip(iss_ids[0], comm_ids[0], sim_scores.data.tolist(), plabels.data.tolist(), labels.tolist()):
            plabels_data.append([iss_id, comm_id, sim_score, plabel, label])
    return plabels_data


def evaluate(args, model, eval_dataset):
    stime = datetime.datetime.now()
    eval_output_dir = args.output_dir

    args.seed += 3

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1,
                                 pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
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
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    return result


def test(args, model, tokenizer, stime):
    # Note that DistributedSampler samples randomly
    eval_dataset = TextDataset(tokenizer, args, os.path.join(args.data_dir, args.pro + '_TEST.csv'))
    args.seed += 3
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("********** Running Test **********")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
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
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    print(preds[:25], labels[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'], dtype=object)
    for key in sorted(result.keys()):
        print('-'*10 + "  {} = {}".format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]
    result_dir = os.path.join(args.result_dir, args.pro)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    dfScores.to_csv(os.path.join(result_dir, args.pro + "_Metrics_st.csv"), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'
    logits4class0, logits4class1 = \
        [logits[iclass][0] for iclass in range(len(logits))],\
        [logits[iclass][1] for iclass in range(len(logits))]
    df = pd.DataFrame(np.transpose([issues, commits, logits4class0, logits4class1, preds, labels]),
                      columns=['Issue_Key', 'Commit_SHA', '0_logit', '1_logit', 'preds', 'labels'])
    df.to_csv(os.path.join(result_dir, args.pro + "_Prediction_st.csv"), index=False)


def main(datasets_dir, outputs_dir, result_dir, pro, key, task_type):
    sys.argv.append(f'--data_dir={datasets_dir}')
    sys.argv.append(f'--output_dir={outputs_dir}')
    sys.argv.append(f'--result_dir={result_dir}')
    sys.argv.append(f'--pro={pro}')
    sys.argv.append(f'--key={key}')

    sys.argv.append('--text_model_path=roberta-large')
    sys.argv.append('--code_model_path=codeBERT')
    sys.argv.append('--tokenizer_name=roberta-large')

    sys.argv.append('--max_seq_length=512')
    sys.argv.append('--num_train_epochs=10')
    sys.argv.append('--train_batch_size=4')
    sys.argv.append('--eval_batch_size=4')
    # sys.argv.append('--learning_rate=1r-5')
    # sys.argv.append('--weight_decay=0.0')
    # sys.argv.append('--seed=42')

    print("======BTLink BEGIN...======" * 20)
    stime = datetime.datetime.now()
    args = getargs()
    args.lab_frac = 0.1
    args.train_lab_batch_size = 1
    args.train_unlab_batch_size = 3
    args.p_cutoff = 0.8
    print(args.key)
    print(f'===Project:{args.pro}---{args.pro}==='*5)
    print("device: {}, n_gpu: {}".format(args.device, args.n_gpu) )
    # Set seed
    set_seed(args.seed)

    # # Roberta
    # config = RobertaConfig.from_pretrained(args.text_model_path)
    # config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    # textEncoder = AutoModel.from_pretrained(args.text_model_path, config=config)
    #
    # # CodeBERT
    # config4Code = RobertaConfig.from_pretrained(args.code_model_path)
    # config4Code.num_labels = 2
    # codeEncoder = AutoModel.from_pretrained(args.code_model_path, config=config4Code)
    # model = BTModel(textEncoder, codeEncoder, config.hidden_size, config4Code.hidden_size, args.num_class)
    # model.to(args.device)
    checkpoint_prefix = args.pro + '_checkpoint-best/model_pre.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model = torch.load(output_dir)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(args.device)
    print("Training/evaluation parameters {}".format(args))
    # Training
    if args.do_train:
        # sample - dataset
        train_lab_dataset = Artifacts(tokenizer, args, os.path.join(args.data_dir, args.pro + f'_TRAIN_lab{int(args.lab_frac * 100)}.csv'))
        train_unlab_dataset = Artifacts(tokenizer, args, os.path.join(args.data_dir, args.pro + f'_TRAIN_unlab{int((1 - args.lab_frac) * 100)}.csv'))
        args.seed += 3
        train(args, train_lab_dataset, train_unlab_dataset, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = args.pro + '_checkpoint-best/model_st.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        test(args, model, tokenizer, stime)


if __name__ == "__main__":
    main()
