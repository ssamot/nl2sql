from einops import rearrange
import torch
import transformers
from transformers import T5Tokenizer, T5EncoderModel, T5Config
transformers.logging.set_verbosity_error()
import numpy as np
from datasets import load_dataset
import random, warnings
warnings.filterwarnings("ignore")
import click



np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

# google/t5-v1_1-base

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
            #truncation=True,
            pad_to_max_length = True,
            add_special_tokens=False
            #return_special_tokens_mask = True

        )

        input_ids = encoded.input_ids.to(device)
        attn_mask = encoded.attention_mask.to(device)

        t5.eval()

        with torch.no_grad():
            output = t5(input_ids=input_ids, attention_mask=attn_mask)
            encoded_text = output.last_hidden_state#.detach()

        encoded_text = encoded_text.detach()#.numpy()

        attn_mask = attn_mask.bool()
        #
        encoded_text = encoded_text.masked_fill(
            ~rearrange(attn_mask, '... -> ... 1'),
            0.)  # just force all embeddings that is padding to be equal to 0.
        #encoded_text[:,:,:] = attn_mask[:,:,np.newaxis] * encoded_text[:,:,:]

        return encoded_text.numpy()


    def t5_tokenize_text(self, texts):
        t5, tokenizer = self.model, self.tokenizer

        if torch.cuda.is_available():
            t5 = t5.cuda()

        device = next(t5.parameters()).device

        encoded = tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            #max_length=self.MAX_LENGTH,
            #truncation=True,
            pad_to_max_length = True,
            add_special_tokens=False
            #return_special_tokens_mask = True

        )

        input_ids = encoded.input_ids.to(device)

        return input_ids.detach().numpy()


    def get_tokenizer(self, name):
        tokenizer = T5Tokenizer.from_pretrained(name, local_files_only=False)
        return tokenizer


    def get_model(self, name):
        model = T5EncoderModel.from_pretrained(name, local_files_only=False)
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
            enc_head.extend(comma)
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

def encode_conds(sql, questions, table, comma,  Tfive):
    all_encs = []
    maximum_length = -1
    for i in range(len(sql)):
        # start = [1]
        # end = [2]
        # #comma = [3]
        # dash = [4]
        # offset = 10

        ci = sql[i]["conds"]["column_index"]
        oi = sql[i]["conds"]["operator_index"]
        co = sql[i]["conds"]["condition"]
        sel_index = [sql[i]["sel"]]
        agg_index = [sql[i]["agg"]]


        #sel_text = [table[i]["header"][sel_index[0]]]
        #agg_text = [table[i]["header"][agg_index[0]]]

        #print(sel_index)
        #print(table[i]["header"][sel_index[0]])
        # s = Tfive.t5_tokenize_text( + ", "])[0]
        # a = Tfive.t5_tokenize_text(  + ": "])[0]

        #encode_conds = Tfive.t5_tokenize_text([f"ST,{sel_text},{agg_text}:"])
        #print(encode_conds.shape)
        #print(encode_conds.shape)
        all_encs.append(agg_index)

        #exit()
       # print(encoded_conds)

        #print(sql[i])
        # for j in range(len(co)):
        #     index = questions[i].upper().index(co[j].upper())
        #     # print(index, index+len(co[j].upper()) )

            #index_start = [[int(x) + 1] for x in str(index).zfill(3)]
            #index_end = [[int(x) + 1] for x in
            #             str(index + len(co[j].upper())).zfill(3)]

            #q_splitted = questions[i].upper().split()
            #co_splitted = co[j].upper().split()

            #print(co_splitted)
            #matching = [s for s in q_splitted if co_splitted[0] in s]
            #index = q_splitted.index(matching[0])
            # if(len(matching)!=1):
            #     print(sql[i], questions[i])
            #     print(matching, index, len(matching))
            #     print("=======")

            # q_stripped = re.sub(r'[^A-Za-z0-9 ]+', ' ', questions[i].upper())
            # co_stripped = re.sub(r'[^A-Za-z0-9 ]+', ' ', co[j].upper())
            # q_stripped = " ".join(q_stripped.split())
            # co_stripped = " ".join(co_stripped.split())
            #
            # q_token = list(Tfive.t5_tokenize_text([q_stripped])[0])
            # co_t = Tfive.t5_tokenize_text([co_stripped])[0]# + 20
            #
            #
            # print(questions[i])
            # print(q_token)
            # #print(co_t)
            # #c_dec = [Tfive.tokenizer.decode([c]) for c in co_t ]
            # #print(c_dec)
            # print(co_stripped,"-----",  q_stripped, "stripeed")
            # start = q_token.index(co_t[0])
            # end = q_token.index(co_t[-1])
            # print(start, end)
            # print("=======")
            #exit()
            #co_t = [[c] for c in co_t[0]]
            #print(co_t, co_t.T.shape)
            #exit()

            # encoded_conds.extend([
            #     [ci[j] + offset],
            #     [oi[j] + offset],
            #     [index + offset],
            #     [index + len(co[j].upper()) +  offset]
            #
            # ])

            # encoded_conds.extend(index_start)
            # encoded_conds.append(dash)
            # encoded_conds.extend(index_end)
            # if(j!=len(co)-1):
            #     encoded_conds.append(comma)
        #encoded_conds.append(end)


    return all_encs


@click.command()
@click.argument('output_filepath', type=click.Path(exists=True))
def main(output_filepath):

    Tfive = T5()

    max_length = 100000
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

    # headers = []
    # for d in train_table, validation_table, test_table:
    #     for t in d:
    #         headers.extend(t["header"])
    # headers = list(set(headers))
    #
    # headers_encoded = []
    # for h in headers:
    #     h_e = Tfive.t5_encode_text([h])
    #     headers_encoded.append(h_e[0])
    #
    # header_dict = dict(zip(headers, headers_encoded))

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
        #questions_encoded = pad_sequences(questions_encoded, padding='post')
        #exit()

        #headers_encoded = encode_header(table, header_dict, comma, end)
        #headers_encoded = pad_sequences(headers_encoded, padding='post')





        # headers_questions_encoded = []
        # for i,q in enumerate(questions_encoded):
        #     f = np.concatenate([headers_encoded[i], q], axis = 0)
        #     headers_questions_encoded.append(f)
        #
        #
        # headers_questions_encoded = pad_sequences(headers_questions_encoded,padding='post', maxlen=110)
        # #print(everything.shape)
        #exit()

        output_encoded = encode_conds(sql, questions, table, comma, Tfive)
        output_encoded = pad_sequences(output_encoded, padding='post')

        np.savez(f"{output_filepath}/{dataset}.npz",
                 #questions_encoded = questions_encoded,
                 #headers_encoded = headers_encoded,
                 output_encoded = output_encoded,
                 headers_questions_encoded = questions_encoded )


        print(questions_encoded.shape, "questions_encoded")
        #print(headers_encoded.shape, "headers_encoded")
        print(output_encoded.shape, "output_encoded" )
        #print()




if __name__ == '__main__':
    main()

