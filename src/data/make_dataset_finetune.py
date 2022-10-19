from torch.utils.data import Dataset, DataLoader,IterableDataset
from transformers import T5Tokenizer
import torch
from datasets import load_dataset
import click
from copy import deepcopy

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

        start = "[START_SQL]"
        end   = "[END_SQL]"

        inputs = []

        for h,q in zip(train_table, train_questions):
            columns = h['header']
            #columns = ":".join(columns)
            str_cols = ""
            for i,col in enumerate(columns):
                str_cols += f"col{i}:{col}, "
            #print(str_cols)
            #exit()
            input = f"[START]-{str_cols}---{q}-[END]"
            #print(input)
            inputs.append(input)

        targets = []

        for t in train_sql:
            sel = t["sel"]
            agg = t["agg"]
            conds = f"ci:{t['conds']['column_index']}::oi:{t['conds']['operator_index']}:c{t['conds']['condition']}::"
            target = f"{start} col{sel}:agg{agg}:{conds} {end}"
            #print(target)
            #exit()
            #target = f"{start} {t['human_readable']} {end}"
            targets.append(target)


        if(split == "train"):
            augment = 4
            print(split)

            import numpy as np

            for i in range(augment):
                for h, q, t in zip(train_table, train_questions,train_sql):

                    columns = np.array(h['header'])
                    sample = np.random.choice(len(columns), len(columns), replace=False)

                    columns = columns[sample]



                    str_cols = ""
                    for i, col in enumerate(columns):
                        str_cols += f"col{i}:{col}, "
                    # print(str_cols)
                    # exit()
                    input = f"[START]-{str_cols}---{q}-[END]"
                    inputs.append(input)

                    sample = list(sample)
                    sel = sample.index(t["sel"])
                    agg = t["agg"]

                    new_conds = []
                    for co in t['conds']['column_index']:
                        new_conds.append(sample.index(co))

                    conds = f"ci:{new_conds}::oi:{t['conds']['operator_index']}:c{t['conds']['condition']}::"
                    target = f"{start} col{sel}:agg{agg}:{conds} {end}"

                    #print(input)

                    targets.append(target)



        encoded_inputs = self.tokenizer.batch_encode_plus(
            inputs,
            return_tensors="pt",
            #max_length=self.max_len,
            # truncation=True,
            pad_to_max_length=True,
            add_special_tokens=False
            # return_special_tokens_mask = True
        )

        encoded_targets = self.tokenizer.batch_encode_plus(
            targets,
            return_tensors="pt",
            #max_length=self.max_len,
            # truncation=True,
            pad_to_max_length=True,
            add_special_tokens=False
            # return_special_tokens_mask = True
        )

        self.inputs = encoded_inputs
        self.targets = encoded_targets


        print("inputs", split, self.inputs["input_ids"].shape)
        print("targets", split, self.targets["input_ids"].shape)

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