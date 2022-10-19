import shutil

from tensorflow import keras as keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
from tqdm import trange
import click
from tensorflow.keras.utils import pad_sequences

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
    X_train = data["headers_questions_encoded"]
    Y_train = data["output_encoded"]
    vocab_size = 17
    Y_train = to_categorical(Y_train, vocab_size)
    #print(Y_train)

    data = np.load(f"{data_filepath}/validation.npz", allow_pickle=True)
    X_validation = data["headers_questions_encoded"]
    X_validation = pad_sequences(X_validation, padding='post', maxlen=85)
    Y_validation = data["output_encoded"]

    Y_validation = to_categorical(Y_validation, vocab_size)

    print(X_validation.shape)
    print(Y_validation.shape)


    embed_dim = X_train.shape[-1]
    max_length = X_train.shape[-2]

    print("Embed_dim:", embed_dim, max_length)
    encoded_seq_inputs = keras.layers.Input(shape=(85, embed_dim),
                                            name="encoder_embeddings")

    x = keras.layers.Flatten()(encoded_seq_inputs)
    x = keras.layers.Dense(1000, activation = "relu")(x)
    #x = keras.layers.Masking()(encoded_seq_inputs)
    #x = keras.layers.LSTM(1000)(x)https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/t5.py
    x = keras.layers.Dense(vocab_size, activation = "softmax")(x)

    decoder = keras.Model([encoded_seq_inputs], x)
    decoder.compile(loss="categorical_crossentropy",metrics = ["accuracy"],
                optimizer=keras.optimizers.Adam(), jit_compile=False)

    decoder.fit(X_train, Y_train, validation_data = (X_validation, Y_validation), epochs = 300)
    Y_hat = decoder.predict(X_train)
    print(Y_hat)


if __name__ == '__main__':
    main()