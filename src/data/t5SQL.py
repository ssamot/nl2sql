import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config
transformers.logging.set_verbosity_error()
import numpy as np
from datasets import load_dataset
import random, warnings
warnings.filterwarnings("ignore")
from query import Query


np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


class T5:
    def __init__(self, name = 'google/t5-v1_1-base', max_length = 100):
        self.T5_CONFIGS = {}
        self.MAX_LENGTH = max_length
        self.model, self.tokenizer = self.get_model_and_tokenizer(name)


    def t5_encode_text(self, texts):
        t5, tokenizer = self.model, self.tokenizer

        if torch.cuda.is_available():
            t5 = t5.cuda()

        device = next(t5.parameters()).device

        encoded = tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            max_length=self.MAX_LENGTH,
            truncation=True,
            pad_to_max_length = True,
            add_special_tokens=False
            #return_special_tokens_mask = True
        )

        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)

        t5.eval()

        with torch.no_grad():
            output = t5(input_ids=input_ids, attention_mask=attn_mask)
            encoded_text = output.last_hidden_state.detach()

        encoded_text = encoded_text.detach().numpy()
        encoded_text[:,:,:] = attn_mask[:,:,np.newaxis] * encoded_text[:,:,:]

        return encoded_text


    def get_tokenizer(self, name):
        tokenizer = T5Tokenizer.from_pretrained(name, local_files_only=True)
        return tokenizer


    def get_model(self, name):
        model = T5EncoderModel.from_pretrained(name, local_files_only=True)
        return model


    def get_model_and_tokenizer(self,name):
        T5_CONFIGS = self.T5_CONFIGS

        if name not in T5_CONFIGS:
            T5_CONFIGS[name] = dict()
        if "model" not in T5_CONFIGS[name]:
            T5_CONFIGS[name]["model"] = self.get_model(name)
        if "tokenizer" not in T5_CONFIGS[name]:
            T5_CONFIGS[name]["tokenizer"] = self.get_tokenizer(name)

        return T5_CONFIGS[name]['model'], T5_CONFIGS[name]['tokenizer']


    def get_encoded_dim(self,name):
        T5_CONFIGS = self.T5_CONFIGS
        if name not in T5_CONFIGS:
            # avoids loading the model if we only want to get the dim
            config = T5Config.from_pretrained(name)
            T5_CONFIGS[name] = dict(config=config)
        elif "config" in T5_CONFIGS[name]:
            config = T5_CONFIGS[name]["config"]
        elif "model" in T5_CONFIGS[name]:
            config = T5_CONFIGS[name]["model"].config
        else:
            assert False
        return config.d_model




# encoding text


if __name__ == '__main__':
    dataset = load_dataset('wikisql')
    train_questions = dataset["train"]["question"][:10]
    #table = dataset["train"]["table"]
    #print(dataset["train"])
    #exit()
    Tfive = T5()
    print(len(train_questions))
    #exit()
    qeustions = Tfive.t5_encode_text(train_questions[:10])
    exit()
    for p in dataset["train"]:
        question = p["question"]
        table = p["table"]
        sql = p["sql"]
        header = table["header"]

        for l in sql:
            print(l, sql[l])
        #print

        print(header)
        print(sql["human_readable"])

        qp = Query.from_dict(p['sql'], header )
        print(qp)
        print("========")
        #exit()
        #except Exception as e:
        #    print(e)

        #print(sql)

