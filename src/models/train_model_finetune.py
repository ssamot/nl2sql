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
from pytorch_lightning.strategies.ddp import DDPStrategy
import numpy as np

# based on this https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb

logger = logging.getLogger(__name__)

class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams.update(hparams)
        # for key in hparams.keys():
        #     self.hparams[key] = hparams[key]

        #self.save_hyperparameters()

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path)

        #self.model.encoder.requires_grad = False
        # for p in self.model.encoder.parameters():
        #     p.requires_grad = False
        #print(self.model.decoder.forward.__code__.co_varnames)

        #decoder_layer = torch.nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)

        #self.model.decoder = torch.nn.TransformerDecoder(decoder_layer, 1)
        #print(self.model.decoder.forward.__code__.co_varnames)
        #exit()
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())

        params = sum([np.prod(p.size()) for p in model_parameters])
        print(self.model)
        print("Trainable params", params)
        #from torchsummary import summary
        #exit()
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



    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.model.save_pretrained(f"{self.hparams.model_filepath}/model_t5.model")
        self.log("avg_val_loss", avg_loss, sync_dist=True)

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        #self.log("avg_train_loss", avg_train_loss,sync_dist=True)
        return None

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

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
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

@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):

    train_dataset = Wikidataset(data_filepath, "train", False)
    validation_dataset = Wikidataset(data_filepath, "validation", False)

    args = dict(
        data_dir=data_filepath,  # path for data files
        model_filepath=model_filepath,  # path to save the checkpoints
        train_dataset = train_dataset,
        validation_dataset = validation_dataset,
        model_name_or_path='t5-small',
        tokenizer_name_or_path='t5-small',
        max_seq_length=512,
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=64,
        eval_batch_size=64,
        num_train_epochs=10,
        gradient_accumulation_steps=16,
        n_gpu=2,
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
        max_epochs=args["num_train_epochs"],
        #early_stop_callback=False,
        precision=16 if args["fp_16"] else 32,
        #amp_level=args["opt_level"],
        gradient_clip_val= args["max_grad_norm"],
        #checkpoint_callback=checkpoint_callback,
        #callbacks=[LoggingCallback()],
        accelerator='gpu', devices=2,
        logger = CSVLogger("./logs"),
        strategy = DDPStrategy(find_unused_parameters=False),
    )


    trainer = pl.Trainer(**train_params)
    trainer.fit(model)
    model.model.save_pretrained(f"{model_filepath}/model_t5.model")


if __name__ == '__main__':
    main()
