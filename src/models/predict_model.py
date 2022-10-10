import numpy as np
from datasets import load_dataset
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
import click
import tensorflow.keras as keras

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

    TART_TOKEN = "[START]"
    END_TOKEN = "[END]"

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

    print("QUESTION:", train_questions[0])
    print("TABLE:", train_table[0]["header"])
    print("SQL:",train_sql[0])

    decode_single_sequence(X_train[0:1],nn)




def decode_single_sequence(input_sentence,decoder, max_decoded_sentence_length=110):
    # print(X_features[0][0].shape, "oirginal feature")

    #X_features = TFive.t5_encode_text([input_sentence])[:,:-1,:]


    start = [11]
    end = [12]
    comma = [13]

    decoded_sentence = np.zeros(shape = (1,100,1))
    decoded_sentence[1][0][1] = start[0]

    for i in range(0, max_decoded_sentence_length):
        print(decoded_sentence)

        #Y = Y[:, :-1]
        predictions = decoder.predict([input_sentence, decoded_sentence], verbose=False)
        sampled_token_index = np.argmax(predictions[0, i, :], axis=-1)
        print(sampled_token_index)
        exit()

        #if(predictions == )


        #if decoded_sentence.endswith("[END]"):
        #    break

    print(decoded_sentence)
    return decoded_sentence


    


if __name__ == '__main__':
    main()