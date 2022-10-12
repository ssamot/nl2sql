from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import torch
from datasets import load_dataset

class Wikidataset(Dataset):
    def __init__(self, cache_filepath, split,  build,  name ="t5-small", max_len=512):


        self.tokenizer = T5Tokenizer.from_pretrained(name, local_files_only=False)

        self.max_len = max_len

        self.inputs_file = f"{cache_filepath}/{split}_inputs.npz"
        self.targets_file = f"{cache_filepath}/{split}_targets.npz"

        if(build):
            self._build(split)
        else:
            self.inputs = torch.load(self.inputs_file)
            self.targets = torch.load(self.targets_file)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index][
            "attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index][
            "attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build(self, split):
        max_length = 100000
        dataset = load_dataset('wikisql')

        train_questions = dataset[split]["question"][:max_length]
        train_table = dataset[split]["table"][:max_length]
        train_sql = dataset[split]["sql"][:max_length]

        start = "<s>"
        end   = "</s"


        targets = []
        encoded_inputs = self.tokenizer.batch_encode_plus(
            train_questions,
            return_tensors="pt",
            max_length=self.max_len,
            # truncation=True,
            pad_to_max_length=True,
            add_special_tokens=False
            # return_special_tokens_mask = True
        )

        self.inputs = encoded_inputs
        self.targets = targets