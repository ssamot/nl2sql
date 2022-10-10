import numpy as np
from datasets import load_dataset
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
import click
import tensorflow.keras as keras
from data.query import Query

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)



@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):

    data = np.load(f"{data_filepath}/train.npz", allow_pickle=True)
    X_train = data["headers_questions_encoded"]
    Y_train = data["output_encoded"]

    data = np.load(f"{data_filepath}/test.npz", allow_pickle=True)
    X_test = data["headers_questions_encoded"]
    Y_test = data["output_encoded"]

    dataset = load_dataset('wikisql')

    train_questions = dataset["train"]["question"]
    validation_questions = dataset["validation"]["question"]
    test_questions = dataset["test"]["question"]

    train_table = dataset["train"]["table"]
    validation_table = dataset["validation"]["table"]
    test_table = dataset["test"]["table"]

    train_sql = dataset["train"]["sql"]
    validation_sql = dataset["validation"]["sql"]
    test_sql = dataset["test"]["sql"]

    nn = keras.models.load_model(f"{model_filepath}/model.keras", custom_objects=
    {
        "TransformerDecoder": TransformerDecoder,
        "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    })


    example = 1
    print("QUESTION:", train_questions[example])
    print("TABLE:", train_table[example]["header"])
    print("SQL:",train_sql[example])

    #print("Real value", Y_train[example].T[0])
    encoded_real = code2query(Y_train[example].T[0], train_table[example]["header"],
               train_questions[example])
    print("Encoded ground truth:", encoded_real)
    y_hat = decode_single_sequence(X_train[example:example+1],nn)
    encoded_predicted = code2query(y_hat, train_table[example]["header"],
               train_questions[example])
    print("Encoded predicted:", encoded_predicted)

def code2query(code, table, question):
    offset = 10

    start = [1]
    end = [2]

    #q = Query()
    sel = int(code[1]) - offset
    agg = int(code[2]) - offset

    column_index = []
    operator_index = []
    condition = []

    c = code[3]
    for i in range(3, 120, 4):
        if(int(code[i]) == end[0]):
            break
        column_index.append(code[i] - offset)
        operator_index.append(code[i+1] - offset)
        start_str = int(code[i+2]) - offset
        end_str  = int(code[i+3]) - offset
        condition.append(question[start_str:end_str])

    d = {}
    d["column_index"] = column_index
    d["operator_index"] = operator_index
    d["condition"] = condition

    #print(d)

    qr = Query(sel_index = sel, agg_index=agg, conditions=d, columns=table )
    return qr





def decode_single_sequence(input_sentence,decoder, max_decoded_sentence_length=95):
    # print(X_features[0][0].shape, "oirginal feature")

    #X_features = TFive.t5_encode_text([input_sentence])[:,:-1,:]

    start = [1]
    end = [2]



    decoded_sentence = np.zeros(shape = (1,100,1))
    decoded_sentence[0][0][0] = start[0]
    final_dec = [start[0]]



    for i in range(0, max_decoded_sentence_length):
        predictions = decoder.predict([input_sentence, decoded_sentence], verbose=False)
        sampled_token_index = np.argmax(predictions[0, i, :], axis=-1)
        decoded_sentence[0][i+1][0] = sampled_token_index
        #print(i, sampled_token_index)

        if(sampled_token_index in start + end ):
            final_dec.append(sampled_token_index)
        else:
            final_dec.append(sampled_token_index)


        if(sampled_token_index == end[0]):
            break

    #print(final_dec)
    #print(decoded_sentence[0])
    return final_dec


    


if __name__ == '__main__':
    main()