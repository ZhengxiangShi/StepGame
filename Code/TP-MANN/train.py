import os
import random
import json
import shutil
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
from data_preprocess.preprocess import parse
from model.tp_mann import Tpmann
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(config,
          serialization_path,
          eval_test,
          train_size, 
          force=False):

    dir_path = Path(serialization_path)
    print(dir_path)
    if dir_path.exists() and force:
        shutil.rmtree(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=False)
    model_path = dir_path / "model.pt"
    config_path = dir_path / "config.json"
    writer = SummaryWriter(log_dir=str(dir_path))

    # Read config
    data_config = config["data"]
    trainer_config = config["trainer"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    
    if eval_test:
        task_ids = range(1, 11)
    else:
        task_ids = range(1, 6)
    train_task_ids = range(1, 6)

    word2id_path = os.path.split(data_config['data_path'])[0]

    try:
        with open(os.path.join(word2id_path, 'word2id.json'), 'r') as f_word2id:
            word2id = json.load(f_word2id)
            print('Read existing vocab dictionary.')
    except Exception:
        word2id = None
        print('No word2id file found.')

    train_data_loaders = {}
    valid_data_loaders = {}
    test_data_loaders = {}

    num_train_batches = num_valid_batches = num_test_batches = 0
    max_seq = 0
    for i in task_ids:
        train_raw_data, valid_raw_data, test_raw_data, word2id = parse(data_config["data_path"],
                                                                       str(i), word2id=word2id,
                                                                       use_cache=True, cache_dir_ext="")
        train_raw_data = list(train_raw_data)
        train_raw_data[0] = train_raw_data[0][:train_size]
        train_raw_data[1] = train_raw_data[1][:train_size]
        train_raw_data[2] = train_raw_data[2][:train_size]
        train_raw_data[3] = train_raw_data[3][:train_size]

        valid_epoch_size = valid_raw_data[0].shape[0]
        test_epoch_size = test_raw_data[0].shape[0]

        max_seq = max(max_seq, train_raw_data[0].shape[2])
        valid_batch_size = valid_epoch_size // 73
        test_batch_size = test_epoch_size // 73

        train_dataset = TensorDataset(*[torch.LongTensor(a) for a in train_raw_data[:-1]])
        valid_dataset = TensorDataset(*[torch.LongTensor(a) for a in valid_raw_data[:-1]])
        test_dataset = TensorDataset(*[torch.LongTensor(a) for a in test_raw_data[:-1]])

        train_data_loader = DataLoader(train_dataset, batch_size=trainer_config["batch_size"], shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=valid_batch_size)
        test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        train_data_loaders[i] = [iter(train_data_loader), train_data_loader]
        valid_data_loaders[i] = valid_data_loader
        test_data_loaders[i] = test_data_loader

        num_train_batches += len(train_data_loader)
        num_valid_batches += len(valid_data_loader)
        num_test_batches += len(test_data_loader)

    with open(os.path.join(word2id_path, 'word2id.json'), 'w') as f_word2id:
        json.dump(word2id, f_word2id)

    print(f"total train data: {num_train_batches*trainer_config['batch_size']}")
    print(f"total valid data: {num_valid_batches * valid_batch_size}")
    print(f"total test data: {num_test_batches * test_batch_size}")
    print(f"vocab size {len(word2id)}")
    print('There are {} tasks in total'.format(len(train_data_loaders)))

    model_config["vocab_size"] = len(word2id)
    model_config["max_seq"] = max_seq
    model_config["symbol_size"] = model_config['symbol_size']
    trainer_config['train_size'] = train_size

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Tpmann(model_config).to(device)
    
    print('\n\n')
    count_parameters(model)
    print('\n\n')
    
    optimizer = optim.Adam(model.parameters(),
                           lr=optimizer_config["lr"], betas=(optimizer_config["beta1"], optimizer_config["beta2"]))

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 230, 250], gamma=0.5)

    max_acc = 0

    if not eval_test:
        with config_path.open("w") as fp:
            json.dump(config, fp, indent=4)
    if eval_test:
        print(f"testing ... load {model_path.absolute()}")
        model.load_state_dict(torch.load(model_path.absolute()))
        # Evaluation on test data
        model.eval()
        correct = 0
        test_loss = 0
        with torch.no_grad():
            total_test_samples = 0
            single_task_acc = [0] * len(test_data_loaders)
            for k, te in test_data_loaders.items():
                test_data_loader = te
                task_acc = 0
                single_task_samples = 0
                for story, story_length, query, answer in tqdm(test_data_loader):
                    logits = model(story.to(device), query.to(device))
                    answer = answer.to(device)
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    task_acc += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    test_loss += loss.sum().item()
                    total_test_samples += story.shape[0]
                    single_task_samples += story.shape[0]
                print(f"validate acc task {k}: {task_acc/single_task_samples}")
                single_task_acc[k - 1] = task_acc/single_task_samples
            test_acc = correct / total_test_samples
            test_loss = test_loss / total_test_samples
        print(f"Test accuracy: {test_acc:.3f}, loss: {test_loss:.3f}")
        print(f"test avg: {np.mean(single_task_acc)}")
        raise True

    # model.load_state_dict(torch.load(model_path.absolute()))
    for i in range(trainer_config["epochs"]):
        logging.info(f"##### EPOCH: {i} #####")
        # Train
        model.train()
        correct = 0
        train_loss = 0
        torch.cuda.synchronize()
        for _ in tqdm(range(num_train_batches)):
            # if _ == 10: break
            loader_i = random.randint(0, len(train_task_ids)-1)+1
            try:
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            except StopIteration:
                train_data_loaders[loader_i][0] = iter(train_data_loaders[loader_i][1])
                story, story_length, query, answer = next(train_data_loaders[loader_i][0])
            optimizer.zero_grad()
            logits = model(story.to(device), query.to(device))
            answer = answer.to(device)
            correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
            correct += correct_batch.item()

            loss = loss_fn(logits, answer)
            train_loss += loss.sum().item()
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_gradient_norm"])
            # nn.utils.clip_grad_value_(model.parameters(), 10)

            optimizer.step()
        
        torch.cuda.synchronize()
        train_acc = correct / (num_train_batches*trainer_config["batch_size"])
        train_loss = train_loss / (num_train_batches*trainer_config["batch_size"])

        # Validation
        model.eval()
        correct = 0
        valid_loss = 0
        with torch.no_grad():
            total_valid_samples = 0
            for k, va in valid_data_loaders.items():
                valid_data_loader = va
                task_acc = 0
                single_valid_samples = 0
                for story, story_length, query, answer in valid_data_loader:
                    logits = model(story.cuda(), query.cuda())
                    answer = answer.cuda()
                    correct_batch = (torch.argmax(logits, dim=-1) == answer).sum()
                    correct += correct_batch.item()
                    loss = loss_fn(logits, answer)
                    valid_loss += loss.sum().item()
                    task_acc += correct_batch.item()
                    total_valid_samples += story.shape[0]
                    single_valid_samples += story.shape[0]
                print(f"validate acc task {k}: {task_acc/single_valid_samples}")
                writer.add_scalars("valid_acc_task_{}".format(k), {"valid": task_acc/single_valid_samples}, i)
            valid_acc = correct / total_valid_samples
            valid_loss = valid_loss / total_valid_samples
            if valid_acc > max_acc:
                print(f"saved model...{model_path}")
                torch.save(model.state_dict(), model_path.absolute())
                max_acc = valid_acc

        writer.add_scalars("accuracy", {"train": train_acc,
                                        "validation": valid_acc}, i)
        writer.add_scalars("loss", {"train": train_loss,
                                    "validation": valid_loss}, i)

        logging.info(f"\nTrain accuracy: {train_acc:.3f}, loss: {train_loss:.3f}"
                     f"\nValid accuracy: {valid_acc:.3f}, loss: {valid_loss:.3f}")
        # if optimizer_config.get("decay", False) and valid_loss < optimizer_config["decay_thr"] and not decay_done:
        #     scheduler.decay_lr(optimizer_config["decay_factor"])
        #     decay_done = True

        scheduler.step()
        print('Current_learning_rate:', get_lr(optimizer))

    writer.close()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--config-file", type=str, metavar='PATH', default="config.json",
                        help="Path to the model config file")
    parser.add_argument("--path", type=str, metavar='PATH', default="./default_path/",
                        help="Serialization directory path")
    parser.add_argument("--eval-test", default=False, action='store_true',
                        help="Whether to eval model on test dataset after training (default: False)")
    parser.add_argument("--train_size", default=10000, type=int)
    parser.add_argument("--logging-level", type=str, metavar='LEVEL', default=20, choices=range(10, 51, 10),
                        help="Logging level (default: 20)")
    args = parser.parse_args()

    logging.basicConfig(level=args.logging_level)

    seed_torch(args.seed)
    print('Use seed: {}'.format(args.seed))

    with open(args.config_file, "r") as fp:
        config = json.load(fp)

    train(config, args.path, args.eval_test, args.train_size)
