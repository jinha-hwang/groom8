from typing import List, Dict, Any, Sequence
from collections import Counter
from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .datasets import TokenizedKoMRC


class Tokenizer:
    def __init__(self,
        id2token: List[str], 
        max_length: int=1024,
        pad: str='<pad>', unk: str='<unk>', cls: str='<cls>', sep: str='<sep>'
    ):
        self.pad = pad
        self.unk = unk
        self.cls = cls
        self.sep = sep
        self.special_tokens = [pad, unk, cls, sep]

        self.max_length = max_length

        self.id2token = self.special_tokens + id2token
        self.token2id = {token: token_id for token_id, token in enumerate(self.id2token)}

    @property
    def vocab_size(self):
        return len(self.id2token)
    
    @property
    def pad_id(self):
        return self.token2id[self.pad]
    @property
    def unk_id(self):
        return self.token2id[self.unk]
    @property
    def cls_id(self):
        return self.token2id[self.cls]
    @property
    def sep_id(self):
        return self.token2id[self.sep]

    @classmethod
    def build_vocab(cls,
        dataset: TokenizedKoMRC, 
        min_freq: int=5
    ):
        counter = Counter(chain.from_iterable(
            sample['context'] + sample['question']
            for sample in tqdm(dataset, desc="Counting Vocab")
        ))

        return cls([word for word, count in counter.items() if count >= min_freq])
    
    def decode(self,
        token_ids: Sequence[int]
    ):
        return [self.id2token[token_id] for token_id in token_ids]

    def sample2ids(self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        context = [self.token2id.get(token, self.unk_id) for token in sample['context']]
        question = [self.token2id.get(token, self.unk_id) for token in sample['question']]

        context = context[:self.max_length-len(question)-3]             # Truncate context
        
        input_ids = [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)

        if sample['answers'] is not None:
            answer = sample['answers'][0]
            start = min(answer['start'] + len(question) + 2, self.max_length - 1)
            end = min(answer['end'] + len(question) + 2, self.max_length - 1)
        else:
            start = None
            end = None

        return {
            'guid': sample['guid'],
            'context': sample['context_original'],
            'question': sample['question_original'],
            'position': sample['context_position'],
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'start': start,
            'end': end
        }

    def logits2answer(self,
        sample: Dict[str, torch.Tensor],
        start_logit: torch.Tensor,         # Shape: (Sequence Length, )
        end_logit: torch.Tensor            # Shape: (Sequence Length, )
    ) -> List[str]:
        position = sample['position']
        start_prob = start_logit[sample['token_type_ids'].bool()][1:-1].softmax(-1)
        end_prob = end_logit[sample['token_type_ids'].bool()][1:-1].softmax(-1)
        probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
        index = torch.argmax(probability).cpu()

        start = index // len(end_prob)
        end = index % len(end_prob)

        start = position[start][0]
        end = position[end][1]

        return sample['context'][start:end]


class TokenizerWrapperDataset:
    def __init__(self, dataset: TokenizedKoMRC, tokenizer: Tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.tokenizer.sample2ids(self.dataset[index])
        sample['attention_mask'] = [1] * len(sample['input_ids'])

        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            sample[key] = torch.tensor(sample[key], dtype=torch.long)

        return sample


class Collator:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = {
            key: [sample[key] for sample in samples]
            for key in samples[0]
        }

        for key in 'start', 'end':
            if samples[key][0] is None:
                samples[key] = None
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)
        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            samples[key] = pad_sequence(
                samples[key], batch_first=True, padding_value=self.tokenizer.pad_id
            )

        return samples