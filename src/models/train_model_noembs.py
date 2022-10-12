import shutil

from tensorflow import keras as keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
from tqdm import trange
import click

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):
    latent_dim = 256
    num_heads = 8
    batch_size = 128

    # from decode import get_max_size


    data = np.load(f"{data_filepath}/train.npz", allow_pickle=True)

    X_train = data["questions_encoded"]
    X_headers = data["headers_encoded"]
    Y_train = data["output_encoded"]


    # data = np.load(f"{data_filepath}/validation.npz", allow_pickle=True)
    # X_validation = data["headers_questions_encoded"]
    # Y_validation = data["output_encoded"]

    # print(X_validation.shape)
    # print(Y_validation.shape)

    #vocab_size = np.max(Y_train) + 1
    #print(vocab_size); exit()
    #vocab_size = 300
    #print(vocab_size)
    print("X_features", X_train.shape)
    print("Y", Y_train.shape)
   # Y_output_train = to_categorical(Y_train, vocab_size)
    #Y_output_validation = to_categorical(Y_validation, vocab_size)
    #print("Y_output", Y_output_train.shape)

    embed_dim = X_train.shape[-1]
    max_length = X_train.shape[-2]

    print("Embed_dim:", embed_dim, max_length)
    encoded_seq_inputs = keras.layers.Input(shape=(None, embed_dim),
                                            name="encoder_embeddings")

    decoder_seq_inputs = keras.layers.Input(shape=(None, embed_dim),
                                            name="decoder")



    timesteps = X_train.shape[1]
    features = embed_dim
    x_encoder = keras.layers.Masking(mask_value=0.,
                                     input_shape=(timesteps, features))(
        encoded_seq_inputs)


    x = TransformerDecoder(intermediate_dim=latent_dim,
                           num_heads=num_heads,
                           dropout=0.2
                           )(decoder_seq_inputs, x_encoder)
    # x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation = "relu")(x)
    decoder_outputs = keras.layers.Dense(512, activation="linear")(x)

    decoder = keras.Model([encoded_seq_inputs, decoder_seq_inputs], decoder_outputs)

    decoder.compile(loss="mse",
                    optimizer="adam", jit_compile=False)
    print(decoder.summary())

    # decoder.fit(data_tr, epochs=2000)
    epochs = 10000
    X_tr = [X_train, X_headers]
    Y_tr = Y_train

    # X_val = [X_validation, Y_validation[:, :-1]]
    # Y_val = Y_output_validation[:, 1:, :]
    #print(Y_val.shape)

    decoder.fit(X_tr, Y_tr,  batch_size = batch_size, epochs=epochs)

    decoder.save(f"{model_filepath}/mdl_tmp.keras")
    shutil.move(f"{model_filepath}/mdl_tmp.keras",
                f"{model_filepath}/model.keras")
    with trange(epochs) as t:
        for i in t:
            # loss = decoder.fit(data_tr, batch_size=64, epochs=1, verbose = False)
            sample = np.random.choice(len(X_train), batch_size)
            # print(sample)

            loss = decoder.train_on_batch(
                [X_train[sample], Y_train[sample][:, :-1]],
                Y_output_train[sample][:, 1:, :])

            #print(loss)

            if (i % 10 == 0):
                decoder.save(f"{model_filepath}/mdl_tmp.keras")
                shutil.move(f"{model_filepath}/mdl_tmp.keras",
                            f"{model_filepath}/model.keras")
                #shutil.os.remove(f"{model_filepath}/mdl_tmp.keras")

            t.set_description(
                'Iter %i, acc %.3f' % (i, loss[-1]))


if __name__ == '__main__':
    main()