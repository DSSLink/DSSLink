import os
import random
import sys
import logging
import argparse
import datetime

from torch.utils.data import DataLoader
from tqdm import trange, tqdm

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_linear_schedule_with_warmup

from common.models import BertClassifier
from common.utils import save_check_point, load_check_point, write_tensor_board, set_seed, evaluate_retrival, evaluate_classification
from common.utils import format_batch_input
from common.read_data import load_examples

logger = logging.getLogger(__name__)


def main(datasets_dir, outputs_dir, dname, group, lab_dtype, unlab_dtype, epochs, task_type=None):
    sys.argv.append(f'--data_dir={datasets_dir}/{dname}/{group}')
    sys.argv.append(f'--output_dir={outputs_dir}/{dname}/{group}')
    sys.argv.append(f'--model_path={outputs_dir}/{dname}/{group}/model_lab{lab_dtype}_epochs{epochs}/final_model')
    sys.argv.append('--code_bert=/home/zhu/projects/codeBERT')
    if task_type is None:
        sys.argv.append(f'--exp_name=model_lab{lab_dtype}_st_epochs{epochs}')
    else:
        sys.argv.append(f'--exp_name=model_lab{lab_dtype}_st_{task_type}_epochs{epochs}')

    sys.argv.append('--logging_steps=50')
    sys.argv.append('--save_steps=5000')
    sys.argv.append('--gradient_accumulation_steps=16')
    sys.argv.append(f'--num_train_epochs={epochs}')
    sys.argv.append('--learning_rate=4e-5')
    sys.argv.append('--valid_step=1000')
    sys.argv.append('--neg_sampling=random')

    sys.argv.append('--batch_size=4')
    sys.argv.append('--batch_size_l=1')
    sys.argv.append('--batch_size_u=3')
    sys.argv.append('--batch_size_e=4')
    sys.argv.append('--p_cutoff=0.8')

    args = get_train_args()
    model = init_train_env(args)
    train_lab_examples = load_examples(os.path.join(args.data_dir, f'train_lab{lab_dtype}'), model)
    num_limit, pairing_rule, use_new_links = assign_task(task_type, len(train_lab_examples))
    train_unlab_examples = load_examples(os.path.join(args.data_dir, f'train_unlab{unlab_dtype}'), model, use_new_links=use_new_links, num_limit=num_limit)
    valid_examples = load_examples(os.path.join(args.data_dir, 'valid'), model)
    # train_dataloader, train_unlab_dataloader, train_dataset = load_unlabeled_data(train_lab_examples, train_unlab_examples, args.batch_size, args.labeled_batch_size)
    train(args, train_lab_examples, train_unlab_examples, valid_examples, model, pairing_rule)
    logger.info('Training finished')

    return args.exp_name

def assign_task(task_type, train_lab_examples_num):
    if task_type is None:
        num_limit = None
        pairing_rule = None
        use_new_links = True
    elif task_type == 'partial':
        num_limit = train_lab_examples_num
        pairing_rule = None
        use_new_links = False
    else:
        assert task_type == 'random'
        num_limit = None
        pairing_rule = 'random'
        use_new_links = True
    return num_limit, pairing_rule, use_new_links

def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir. Should contain the .json files for the task.")
    parser.add_argument("--model_path", default=None, type=str, help="path of checkpoint and trained model, if none will do training from scratch")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--valid_num", type=int, default=100, help="number of instances used for evaluating the checkpoint performance")
    parser.add_argument("--valid_step", type=int, default=50, help="obtain validation accuracy every given steps")

    parser.add_argument("--train_num", type=int, default=None, help="number of instances used for training")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the cached data")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model checkpoints and predictions will be written.", )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--exp_name", type=str, help="name of this execution")
    parser.add_argument("--hard_ratio", default=0.5, type=float, help="The ration of hard negative examples in a batch during negative sample mining")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--neg_sampling", default='random', choices=['random', 'online', 'offline'], help="Negative sampling strategy we apply for constructing dataset. ")
    # parser.add_argument("--code_bert", default='microsoft/codebert-base', choices=['microsoft/codebert-base', 'huggingface/CodeBERTa-small-v1', 'codistai/codeBERT-small-v2'], help="Negative sampling strategy we apply for constructing dataset. ")
    parser.add_argument("--code_bert", default='microsoft/codebert-base')
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].See details at https://nvidia.github.io/apex/amp.html",)

    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--batch_size_l", default=4, type=int, help="Batch size per GPU/CPU for training labeled.")
    parser.add_argument("--batch_size_u", default=4, type=int, help="Batch size per GPU/CPU for training unlabeled.")
    parser.add_argument("--batch_size_e", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--p_cutoff', type=float, default=0.9)

    args = parser.parse_args()
    return args

def init_train_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args.seed, args.n_gpu)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model = BertClassifier(BertConfig(), args.code_bert)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    return model

def get_optimizer_scheduler(args, model, train_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=train_steps
    )
    return optimizer, scheduler

def log_train_info(args, lab_example_num, unlab_example_num, train_steps):
    logger.info("***** Running training *****")
    logger.info("  Num labeled example = %d", lab_example_num)
    logger.info("  Num unlabeled example = %d", unlab_example_num)
    logger.info("  Batch size labeled example = %d", args.batch_size_l)
    logger.info("  Batch size unlabeled example = %d", args.batch_size_u)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", train_steps)

def get_exp_name(args):
    exp_name = "{}_{}_{}_{}"
    time = datetime.datetime.now().strftime("%m-%d %H-%M-%S")

    base_model = ""
    if args.model_path:
        base_model = os.path.basename(args.model_path)
    return exp_name.format('siamese2', args.neg_sampling, time, base_model)

def train(args, train_lab_examples, train_unlab_examples, valid_examples, model, pairing_rule):
    """
    :param args:
    :param train_examples:
    :param valid_examples:
    :param model:
    :param train_iter_method: method use for training in each iteration
    :return:
    """
    if not args.exp_name:
        exp_name = get_exp_name(args)
    else:
        exp_name = args.exp_name

    args.output_dir = os.path.join(args.output_dir, exp_name)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir="./runs/{}".format(exp_name))

    if pairing_rule == 'random':
        unlab_examples = train_unlab_examples.gen_unlab_examples_by_random()
    else:
        unlab_examples = train_unlab_examples.gen_unlab_examples_by_datetime()
    train_unlab_dataloader = DataLoader(unlab_examples, batch_size=args.batch_size)
    lab_examples_num = len(train_lab_examples) * 2
    unlab_examples_num = len(unlab_examples)

    args.train_batch_size = args.batch_size * max(1, args.n_gpu)
    lab_epoch_batch_num = int(lab_examples_num / args.batch_size_l)
    unlab_epoch_batch_num = int(unlab_examples_num / args.batch_size_u)
    args.epoch_batch_num = lab_epoch_batch_num
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (args.epoch_batch_num // args.gradient_accumulation_steps) + 1
    else:
        if args.epoch_batch_num < args.gradient_accumulation_steps:
            args.gradient_accumulation_steps = args.epoch_batch_num
        t_total = args.epoch_batch_num // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer, scheduler = get_optimizer_scheduler(args, model, t_total)
    criterion = torch.nn.CrossEntropyLoss()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # Train!
    log_train_info(args, lab_examples_num, unlab_examples_num, t_total)

    args.global_step = 0
    args.epochs_trained = 0
    args.steps_trained_in_current_epoch = 0

    # Load the model from Stage 1
    model_path = os.path.join(args.model_path, 't_bert.pt')
    model.load_state_dict(torch.load(model_path))
    model = model.to(args.device)

    skip_n_steps_in_epoch = args.steps_trained_in_current_epoch  # in case we resume training
    model.zero_grad()
    train_iterator = trange(args.epochs_trained, args.num_train_epochs, desc="Epoch", disable=args.local_rank not in [-1, 0])
    step_bar = tqdm(initial=args.epochs_trained, total=t_total, desc="Steps")
    for epoch in train_iterator:
        # Extract features and update the pseudolabels
        p_labels_data = extract_pseudo_labels(train_unlab_dataloader, train_unlab_examples, model)
        train_unlab_examples.update_plabels(p_labels_data, args.p_cutoff)
        train_with_neg_sampling(args, model, train_lab_examples, train_unlab_examples, optimizer, scheduler, tb_writer, step_bar, skip_n_steps_in_epoch, criterion)
        if epoch % (args.num_train_epochs // 4) == 0:
            # step invoke validation
            valid_examples.update_embd(model)
            valid_accuracy, valid_loss = evaluate_classification(valid_examples, model, args.batch_size_e, "evaluation/{}/runtime_eval".format(args.neg_sampling), criterion)
            pk, best_f1, map = evaluate_retrival(model, valid_examples, args.batch_size_e, "evaluation/{}/runtime_eval".format(args.neg_sampling))
            tb_data = {
                "valid_accuracy": valid_accuracy,
                "valid_loss": valid_loss,
                "precision@3": pk,
                "best_f1": best_f1,
                "MAP": map
            }
            write_tensor_board(tb_writer, tb_data, args.global_step)

        args.epochs_trained += 1
        skip_n_steps_in_epoch = 0
        args.steps_trained_in_current_epoch = 0

        if args.max_steps > 0 and args.global_step > args.max_steps:
            break

    model_output = os.path.join(args.output_dir, "final_model")
    save_check_point(model, model_output, args, optimizer, scheduler)
    step_bar.close()
    train_iterator.close()
    if args.local_rank in [-1, 0]:
        tb_writer.close()

def re_sort(data, islabs):
    lab_data = data[islabs == 1]
    unlab_data = data[islabs == 0]
    return torch.cat([lab_data, unlab_data], dim=0)

def train_with_neg_sampling(args, model, train_lab_examples, train_unlab_examples, optimizer, scheduler, tb_writer, step_bar, skip_n_steps, criterion):
    """
    Create training dataset at epoch level.
    """
    model.train()
    tr_loss, tr_ac = 0, 0
    train_lab_dataloader = train_lab_examples.random_neg_sampling_dataloader(args.batch_size_l)
    train_unlab_dataloader = train_unlab_examples.random_neg_sampling_dataloader_by_plabels(args.batch_size_u)
    train_lab_iter = iter(train_lab_dataloader)
    train_unlab_iter = iter(train_unlab_dataloader)
    for step in range(args.epoch_batch_num):
        if skip_n_steps > 0:
            skip_n_steps -= 1
            continue

        try:
            lab_batch = next(train_lab_iter)
        except Exception as e:
            train_lab_iter = iter(train_lab_dataloader)
            lab_batch = next(train_lab_iter)
        try:
            unlab_batch = next(train_unlab_iter)
        except Exception as e:
            train_unlab_iter = iter(train_unlab_dataloader)
            unlab_batch = next(train_unlab_iter)

        lab_inputs = format_batch_input(lab_batch, train_lab_examples, model)
        unlab_inputs = format_batch_input(unlab_batch, train_unlab_examples, model)
        inputs = dict()
        for key in lab_inputs.keys():
            inputs[key] = torch.cat([lab_inputs[key], unlab_inputs[key]], dim=0)
        y_lb = lab_batch[2].to(model.device)
        y_ulb = unlab_batch[2].to(model.device)
        num_lb = y_lb.size(0)

        logits = model(inputs)
        logits_lb = logits[:num_lb]
        logits_ulb = logits[num_lb:]

        sup_loss = criterion(logits_lb.view(-1, 2), y_lb.view(-1))
        unsup_loss = criterion(logits_ulb.view(-1, 2), y_ulb.view(-1))
        lambda_u = 1.0
        loss = sup_loss + lambda_u * unsup_loss

        y_pred = logits.data.max(1)[1]
        labels = torch.cat([y_lb, y_ulb], dim=0).to(model.device)
        tr_ac += y_pred.eq(labels).long().sum().item()

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            try:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        else:
            loss.backward()
        tr_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            args.global_step += 1
            step_bar.update()

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and args.global_step % args.logging_steps == 0:
                tb_data = {
                    'lr': scheduler.get_last_lr()[0],
                    'acc': tr_ac / args.logging_steps / (
                            args.train_batch_size * args.gradient_accumulation_steps),
                    'loss': tr_loss / args.logging_steps
                }
                write_tensor_board(tb_writer, tb_data, args.global_step)
                tr_loss = 0.0
                tr_ac = 0.0

            # Save model checkpoint
            if args.local_rank in [-1, 0] and args.save_steps > 0 and args.global_step % args.save_steps == 1:
                # step invoke checkpoint writing
                ckpt_output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(args.global_step))
                save_check_point(model, ckpt_output_dir, args, optimizer, scheduler)

        args.steps_trained_in_current_epoch += 1
        if args.max_steps > 0 and args.global_step > args.max_steps:
            break

def extract_pseudo_labels(train_unlab_dataloader, train_unlab_examples, model):
    model.eval()
    train_iter = iter(train_unlab_dataloader)
    res = list()
    for step in tqdm(range(len(train_unlab_dataloader)), desc='Extracting Pseudo Labels'):
        batch = next(train_iter)
        nl_ids, pl_ids, labels = batch
        inputs = format_batch_input(batch, train_unlab_examples, model)
        with torch.no_grad():
            logits = model(inputs)
            sim_scores = torch.softmax(logits, dim=-1)
            p_labels = torch.max(sim_scores, dim=-1)[1]
        for nl_id, pl_id, sim_score, p_label, label in zip(nl_ids.tolist(), pl_ids.tolist(), sim_scores.data.tolist(), p_labels.data.tolist(), labels.tolist()):
            res.append([nl_id, pl_id, sim_score, p_label, label])
    return res

def log_write(text):
    log_file = 'log.txt'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a', encoding='utf8') as f:
        f.write('%s %s\n' % (current_time, text))
