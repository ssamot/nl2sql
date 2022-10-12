from torch.utils.data import Dataset, DataLoader,IterableDataset
from transformers import T5Tokenizer
import torch
from datasets import load_dataset
import click

class Wikidataset(Dataset):
    def __init__(self, cache_filepath, split,  build,  name ="t5-small", max_len=512):


        self.tokenizer = T5Tokenizer.from_pretrained(name, local_files_only=False)

        self.max_len = max_len

        self.inputs_file = f"{cache_filepath}/{split}_inputs_tokens.pytorch"
        self.targets_file = f"{cache_filepath}/{split}_targets_tokens.pytorch"

        if(build):
            self._build(split)
        else:
            self.inputs = torch.load(self.inputs_file)
            self.targets = torch.load(self.targets_file)


    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, index):
        source_ids = self.inputs["input_ids"][index].squeeze()
        target_ids = self.targets["input_ids"][index].squeeze()

        src_mask = self.inputs[
            "attention_mask"][index].squeeze()  # might need to squeeze
        target_mask = self.targets[
            "attention_mask"][index].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask,
                "target_ids": target_ids, "target_mask": target_mask}

    def _build(self, split):
        max_length = 1000000
        dataset = load_dataset('wikisql')

        train_questions = dataset[split]["question"][:max_length]
        train_table = dataset[split]["table"][:max_length]
        train_sql = dataset[split]["sql"][:max_length]

        start = "<s>"
        end   = "</s"

        inputs = []

        for h,q in zip(train_table, train_questions):
            inputs.append(f"{h['header']}:{q}::")

        targets = []

        for t in train_sql:
            targets.append(f"{start} {t['human_readable']} {end}")

        encoded_inputs = self.tokenizer.batch_encode_plus(
            inputs,
            return_tensors="pt",
            max_length=self.max_len,
            # truncation=True,
            pad_to_max_length=True,
            add_special_tokens=False
            # return_special_tokens_mask = True
        )

        encoded_targets = self.tokenizer.batch_encode_plus(
            targets,
            return_tensors="pt",
            max_length=self.max_len,
            # truncation=True,
            pad_to_max_length=True,
            add_special_tokens=False
            # return_special_tokens_mask = True
        )

        self.inputs = encoded_inputs
        self.targets = encoded_targets



        torch.save(self.inputs,self.inputs_file)
        torch.save(self.targets,self.targets_file)

@click.command()
@click.argument('output_filepath', type=click.Path(exists=True))
def main(output_filepath):
    for split in ["train", "validation", "test"]:
        print(split)
        ds = Wikidataset(output_filepath, split, True)
        # for i,d in enumerate(ds):
        #     print(i)



if __name__ == '__main__':
    main()