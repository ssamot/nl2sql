import argparse
import os
import logging
import click
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from data.make_dataset_finetune import Wikidataset

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

logger = logging.getLogger(__name__)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        #self.hparams.update(hparams)
        for key in hparams.keys():
            self.hparams[key] = hparams[key]

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.hparams.tokenizer_name_or_path)


    def is_logger(self):
        return True
        #return self.trainer.rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None,
            decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    #
    # def training_epoch_end(self, outputs):
    #     avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #     tensorboard_logs = {"avg_train_loss": avg_train_loss}
    #     return None

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs,
                'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if
                           any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]



    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss),
                     "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = self.hparams.train_dataset
        dataloader = DataLoader(train_dataset,
                                batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=4)

        t_total = (
                (len(dataloader.dataset) // (
                            self.hparams.train_batch_size * max(1,
                                                                self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = self.hparams.validation_dataset
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size,
                          num_workers=4)





class LoggingCallback(pl.Callback):
  def on_validation_end(self, trainer, pl_module):
    logger.info("***** Validation results *****")
    if pl_module.is_logger():
      metrics = trainer.callback_metrics
      # Log results
      for key in sorted(metrics):
        if key not in ["log", "progress_bar"]:
          logger.info("{} = {}\n".format(key, str(metrics[key])))

  def on_test_end(self, trainer, pl_module):
    logger.info("***** Test results *****")

    if pl_module.is_logger():
      metrics = trainer.callback_metrics

      # Log and save results to file
      output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
      with open(output_test_results_file, "w") as writer:
        for key in sorted(metrics):
          if key not in ["log", "progress_bar"]:
            logger.info("{} = {}\n".format(key, str(metrics[key])))
            writer.write("{} = {}\n".format(key, str(metrics[key])))

@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):

    train_dataset = Wikidataset(data_filepath, "train", False)
    validation_dataset = Wikidataset(data_filepath, "validation", False)

    args = dict(
        data_dir=data_filepath,  # path for data files
        output_dir=model_filepath,  # path to save the checkpoints
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        model_name_or_path='t5-small',
        tokenizer_name_or_path='t5-small',
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        eval_batch_size=8,
        num_train_epochs=2,
        gradient_accumulation_steps=16,
        n_gpu=1,
        early_stop_callback=False,
        fp_16=False,
        # if you want to enable 16-bit training then install apex and set this to true
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,
        # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
    )
    # args.update({'data_dir': 'aclImdb', 'output_dir': 't5_imdb_sentiment',
    #                   'num_train_epochs': 2})
    #args = argparse.Namespace(**args_dict)
    #print(args)
    #exit()
    model = T5FineTuner(args)

    train_params = dict(
        accumulate_grad_batches=args["gradient_accumulation_steps"],
        gpus=args["n_gpu"],
        max_epochs=args["num_train_epochs"],
        #early_stop_callback=False,
        precision=16 if args["fp_16"] else 32,
        #amp_level=args["opt_level"],
        gradient_clip_val= args["max_grad_norm"],
        #checkpoint_callback=checkpoint_callback,
        #callbacks=[LoggingCallback()],
    )


    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.model.save_pretrained(f"{model_filepath}/model_t5.model")

    ######## should go into predict

    loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    it = iter(loader)
    batch = next(it)
    print(batch["source_ids"].shape)

    outs = model.model.generate(input_ids=batch['source_ids'],
                                attention_mask=batch['source_mask'],
                                max_length=100)

    dec = [model.tokenizer.decode(ids) for ids in outs]

    texts = [model.tokenizer.decode(ids) for ids in batch['source_ids']]
    targets = [model.tokenizer.decode(ids) for ids in batch['target_ids']]

    import textwrap
    for i in range(32):
        lines = textwrap.wrap("Natural language question:\n%s\n" % texts[i].split("::")[0], width=100)
        print("\n".join(lines))
        print("\nActual SQL: %s" % targets[i].split("/s")[0])
        print("Predicted SQL: %s" % dec[i].split("/s")[0])
        print(
            "=====================================================================\n")

if __name__ == '__main__':
    main()
