import os
# stop the noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import click
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from data.make_dataset_finetune import Wikidataset
from pytorch_lightning.loggers import CSVLogger
from transformers import T5ForConditionalGeneration,T5Tokenizer,get_linear_schedule_with_warmup
#from train_model_finetune import T5FineTuner

@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):

    train_dataset = Wikidataset(data_filepath, "train", False)
    validation_dataset = Wikidataset(data_filepath, "validation", False)

    model = T5ForConditionalGeneration.from_pretrained(f"{model_filepath}/model_t5.model")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")


    loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    it = iter(loader)
    batch = next(it)
    print(batch["source_ids"].shape)

    outs = model.generate(input_ids=batch['source_ids'],
                                attention_mask=batch['source_mask'],
                                max_length=100)

    dec = [tokenizer.decode(ids) for ids in outs]
    texts = [tokenizer.decode(ids) for ids in batch['source_ids']]
    targets = [tokenizer.decode(ids) for ids in batch['target_ids']]

    import textwrap
    for i in range(32):
        lines = textwrap.wrap("Natural language question:\n%s\n" % texts[i].split(":::")[0], width=100)
        print("\n".join(lines))
        print("\nActual SQL: %s" % targets[i].split("/s")[0])
        print("Predicted SQL: %s" % dec[i].split("/s")[0])
        print(
            "=====================================================================\n")

if __name__ == '__main__':
    main()
