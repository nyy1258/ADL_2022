import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import torch.nn as nn

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def train_one_epoch(args, model, dataloader, optimizer, loss_fn):
    
    model.train()
    train_loss = []

    for train in dataloader:
        optimizer.zero_grad()
        prediction = model(train['tokens'].to(args.device))
        loss = loss_fn(prediction, train['tags'].to(args.device))
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    avg_train_loss = sum(train_loss) / len(train_loss)

    return avg_train_loss

@torch.no_grad()
def validation(args, model, dataloader):

    model.eval()
    pred_labels = []
    labels = []
    correct = 0
    acc = 0

    for dev in dataloader:
        pred = model(dev['tokens'].to(args.device))
        

        labels.extend(dev["tags"].tolist())  ## answer
        pred_labels.extend(torch.argmax(pred, dim=1).tolist())

    for i in range(len(pred_labels)):
        if(pred_labels[i]==labels[i]):
            correct += 1

    acc = correct / len(labels)

    return acc

def save_checkpoint(args, model, epoch, acc):

    best_ckp_path = args.ckpt_dir / 'best_model.pt'
    torch.save({'model_state':model.state_dict(), 'acc': acc}, best_ckp_path)
    print('Saved model checkpoints into {}...'.format(best_ckp_path))

def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: crecate DataLoader for train / dev datasets
    train_dataloader = DataLoader(datasets[TRAIN], batch_size = args.batch_size, shuffle = True, collate_fn = datasets[TRAIN].collate_fn)
    dev_dataloader = DataLoader(datasets[DEV], batch_size = args.batch_size, shuffle = False, collate_fn = datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
                      embeddings = embeddings, 
                      hidden_size = args.hidden_size, 
                      num_layers = args.num_layers, 
                      dropout = args.dropout, 
                      bidirectional = args.bidirectional, 
                      num_class = datasets[TRAIN].num_classes,
                      max_len = args.max_len
                      )
    model.to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        avg_loss = train_one_epoch(args, model, train_dataloader, optimizer, loss_fn)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        acc = validation(args, model, dev_dataloader)
        epoch_pbar.set_description(f'loss = {avg_loss}, acc = {acc}')

        if acc > best_acc:
            best_acc = acc
            save_checkpoint(args, model, epoch, acc)



def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # build a directary in ./ckpt/slot/
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)