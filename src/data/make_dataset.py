import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config
transformers.logging.set_verbosity_error()
import numpy as np
from datasets import load_dataset
import random, warnings
warnings.filterwarnings("ignore")
import click
from tensorflow.keras.utils import pad_sequences


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
            #max_length=self.MAX_LENGTH,
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



def encode_header(table, header_dict, comma, end):
    headers_encoded = []
    #print(comma.shape)
    #exit()
    for h in table:
        enc_head = []
        for c in h["header"]:
            enc_head.extend(header_dict[c])
            #enc_head.extend(comma)
        enc_head.extend(end)
        enc_head = np.array(enc_head)
        #print(enc_head.shape)
        headers_encoded.append(enc_head)



    #headers_encoded = np.array(headers_encoded)
    return headers_encoded

# def encode_selagg(sql):
#     selagg_enc = []
#     for k in sql:
#         selagg_enc.append([k["sel"], k["agg"]])
#     selagg_enc = np.array(selagg_enc)
#     return selagg_enc

def encode_conds(sql, questions):
    all_encs = []
    maximum_length = -1
    for i in range(len(sql)):
        start = [11]
        end = [12]
        comma = [13]
        dash = [14]

        ci = sql[i]["conds"]["column_index"]
        oi = sql[i]["conds"]["operator_index"]
        co = sql[i]["conds"]["condition"]

        encoded_conds = [
            start,
            [sql[i]["sel"] + 1],
            [sql[i]["agg"] + 1]
        ]

        for j in range(len(co)):
            index = questions[i].upper().index(co[j].upper())
            # print(index, index+len(co[j].upper()) )

            index_start = [[int(x) + 1] for x in str(index).zfill(3)]
            index_end = [[int(x) + 1] for x in
                         str(index + len(co[j].upper())).zfill(3)]

            encoded_conds.extend([
                [ci[j] + 1],
                [oi[j] + 1], ])
            encoded_conds.extend(index_start)
            encoded_conds.append(dash)
            encoded_conds.extend(index_end)
            if(j!=len(co)-1):
                encoded_conds.append(comma)
        encoded_conds.append(end)
        if (len(encoded_conds) > maximum_length):
            maximum_length = len(encoded_conds)
        all_encs.append(encoded_conds)
    return all_encs


@click.command()
@click.argument('output_filepath', type=click.Path(exists=True))
def main(output_filepath):

    Tfive = T5()

    max_length = 100
    dataset = load_dataset('wikisql')

    train_questions = dataset["train"]["question"][:max_length]
    validation_questions = dataset["validation"]["question"][:max_length]
    test_questions = dataset["test"]["question"][:max_length]

    train_table = dataset["train"]["table"][:max_length]
    validation_table = dataset["validation"]["table"][:max_length]
    test_table = dataset["test"]["table"][:max_length]

    train_sql = dataset["train"]["sql"][:max_length]
    validation_sql = dataset["validation"]["sql"][:max_length]
    test_sql = dataset["test"]["sql"][:max_length]

    headers = []
    for d in train_table, validation_table, test_table:
        for t in d:
            headers.extend(t["header"])
    headers = list(set(headers))

    headers_encoded = []
    for h in headers:
        h_e = Tfive.t5_encode_text([h])
        headers_encoded.append(h_e[0])

    header_dict = dict(zip(headers, headers_encoded))

    map = {
        "train":[train_questions, train_table, train_sql ],
        "validation": [ validation_questions, validation_table, validation_sql],
        "test": [ test_questions, test_table, test_sql]
    }

    comma = Tfive.t5_encode_text([","])[0]
    end = Tfive.t5_encode_text(["hend"])[0]

    for dataset in ["train", "validation", "test"]:

        print(dataset)
        questions, table, sql = map[dataset]
        #exit()
        questions_encoded = Tfive.t5_encode_text(questions)
        #exit()

        headers_encoded = encode_header(table, header_dict, comma, end)
        #headers_encoded = pad_sequences(headers_encoded, padding='post')





        headers_questions_encoded = []
        for i,q in enumerate(questions_encoded):
            f = np.concatenate([headers_encoded[i], q], axis = 0)
            headers_questions_encoded.append(f)

        headers_questions_encoded = pad_sequences(headers_questions_encoded,padding='post')
        #print(everything.shape)
        #exit()

        output_encoded = encode_conds(sql, questions)
        output_encoded = pad_sequences(output_encoded, padding='post',
                                       maxlen=headers_questions_encoded.shape[1])

        np.savez(f"{output_filepath}/{dataset}.npz",
                 #questions_encoded = questions_encoded,
                 #headers_encoded = headers_encoded,
                 output_encoded = output_encoded,
                 headers_questions_encoded = headers_questions_encoded )


        print(headers_questions_encoded.shape, "questions_encoded")
        #print(headers_encoded.shape, "headers_encoded")
        print(output_encoded.shape, "output_encoded" )
        #print()




if __name__ == '__main__':
    main()

