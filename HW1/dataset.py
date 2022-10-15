from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab
import torch



class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch = {}
        batch['id'] = [s['id'] for s in samples]
        
        batch['text'] = [s['text'].split() for s in samples]
        batch['text'] = self.vocab.encode_batch(batch['text'], self.max_len)
        batch['text'] = torch.tensor(batch['text'])

        try:
            batch['intent'] = [self.label2idx(s['intent']) for s in samples]
            batch['intent'] = torch.tensor(batch['intent'])
        except KeyError:
            pass

        return batch
        

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    #ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        batch = {}
        token_list = []
        tag_list = []
        id_list = []
        len_list = []

        for sample in samples:
            token_list.append(sample["tokens"])
            # sample: {'tokens': ['hello', 'can', 'i', 'bok', 'a', 'table', 'please'], 'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O'], 'id': 'train-5749'}
            # total 128 sample
            try:
                tag_list.append([self.label2idx(tag) for tag in sample["tags"]] + [self.label2idx("O") for _ in range(self.max_len - len(sample["tags"]))])
            except KeyError:
                pass

            id_list.append(sample["id"])
            len_list.append(len(sample["tokens"]))
        
        encode_token = self.vocab.encode_batch(token_list, self.max_len)
        # token list = [128][128]
        # tag list = [128][128]

        tokens_tensor = torch.tensor(encode_token , dtype = torch.int)
        tags_tensor = torch.tensor(tag_list , dtype = torch.long) 

        batch['tokens'] = tokens_tensor
        batch['tags'] = tags_tensor
        batch['id'] = id_list
        batch['len'] = len_list

        return batch


