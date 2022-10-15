import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

import csv


def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, collate_fn = dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings = embeddings, 
        hidden_size = args.hidden_size, 
        num_layers = args.num_layers, 
        dropout = args.dropout, 
        bidirectional = args.bidirectional, 
        num_class = dataset.num_classes,
        max_len = args.max_len
        )

    model.to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt['model_state'])

    # TODO: predict dataset
    pred_dict = {}
    for test in test_dataloader:
        with torch.no_grad():
            output = model(test['tokens'].to(args.device))
            pred = torch.argmax(output, 1)
            
            for i in range(len(pred)):
                temp = pred.view(-1).cpu().numpy()
                labels = []
                for j in range(test["len"][i]):
                    labels.append(dataset.idx2label(temp[args.max_len * i + j]))
                pred_dict[test["id"][i]] = " ".join(labels)      

    # TODO: write prediction to file (args.pred_file)
    file = open(args.pred_file, 'w')
    writer = csv.writer(file)
    writer.writerow(['id', 'tags'])
    for index, label in pred_dict.items():
        writer.writerow([index, label])

    file.close()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/best_model.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)