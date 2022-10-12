import shutil

from tensorflow import keras as keras
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras_nlp.layers import TransformerDecoder, TokenAndPositionEmbedding
from tqdm import trange
import click
from tensorflow.keras.utils import pad_sequences
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(data_filepath, model_filepath):
    latent_dim = 1000
    num_heads = 16
    batch_size = 128

    # from decode import get_max_size



    data = np.load(f"{data_filepath}/train.npz", allow_pickle=True)
    X_train = data["headers_questions_encoded"]
    Y_train = data["output_encoded"]#
    #print(Y_train.shape)
    w = compute_sample_weight(class_weight = "balanced", y = Y_train)
    #print(w)
    #exit()
    start_token = np.ones(shape = Y_train.shape, dtype=int)*7
    Y_train = np.concatenate([start_token, Y_train], axis = - 1)[:,:,np.newaxis]
    print(Y_train.shape)


    data = np.load(f"{data_filepath}/validation.npz", allow_pickle=True)
    X_validation = data["headers_questions_encoded"]
    Y_validation = data["output_encoded"]#[:,:,np.newaxis]
    X_validation = pad_sequences(X_validation, padding='post', maxlen=89)
    start_token = np.ones(shape=Y_validation.shape, dtype=int) * 7
    Y_validation = np.concatenate([start_token, Y_validation], axis = - 1)[:,:,np.newaxis]


    print(X_validation.shape)
    print(Y_validation.shape)

    vocab_size = np.max(Y_train) + 1
    #print(vocab_size); exit()

    #print(vocab_size)
    print("X_features", X_train.shape)
    print("Y", Y_train.shape)
    Y_output_train = to_categorical(Y_train, vocab_size)
    Y_output_validation = to_categorical(Y_validation, vocab_size)
    print("Y_output", Y_output_train.shape)

    embed_dim = X_train.shape[-1]
    max_length = X_train.shape[-2]

    print("Embed_dim:", embed_dim, max_length)
    encoded_seq_inputs = keras.layers.Input(shape=(None, embed_dim),
                                            name="encoder_embeddings")

    decoder_inputs = keras.Input(shape=(None,), dtype="int64",
                                 name="decoder_inputs")

    timesteps = X_train.shape[1]
    features = embed_dim
    x_encoder = keras.layers.Masking(mask_value=0.,
                                     input_shape=(timesteps, features))(
        encoded_seq_inputs)

    embedding_layer = TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=timesteps,
        embedding_dim=128,
    )(decoder_inputs)

    x = TransformerDecoder(intermediate_dim=latent_dim,
                           num_heads=num_heads,
                           dropout=0.2
                           )(embedding_layer, x_encoder)
    # x = keras.layers.Dropout(0.5)(x)
    #x = keras.layers.Dense(256, activation = "relu")(x)
    decoder_outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)

    decoder = keras.Model([encoded_seq_inputs, decoder_inputs], decoder_outputs)

    decoder.compile(loss=keras.losses.CategoricalCrossentropy(),
                    optimizer="adam", jit_compile=False,
                    metrics=["accuracy"])
    print(decoder.summary())

    # decoder.fit(data_tr, epochs=2000)
    epochs = 30
    X_tr = [X_train, Y_train[:, :-1]]
    Y_tr = Y_output_train[:, 1:, :]

    X_val = [X_validation, Y_validation[:, :-1]]
    Y_val = Y_output_validation[:, 1:, :]
    #print(Y_val.shape)

    decoder.fit(X_tr, Y_tr, validation_data = (X_val, Y_val),
                batch_size = batch_size,
                sample_weight=w,

                epochs=epochs)

    decoder.save(f"{model_filepath}/mdl_tmp.keras")
    shutil.move(f"{model_filepath}/mdl_tmp.keras",
                f"{model_filepath}/model.keras")

    y_hat = decoder.predict(X_val)


    print(Y_train.shape, Y_tr.shape)
    for i in range(0,10):
        print(y_hat[i])
        print(Y_train[i].T)

    y_true = Y_validation[:,1, :]
    y_pred = np.argmax(y_hat.squeeze(), -1)
    print(y_true.shape, y_pred.shape)
    print(classification_report(y_true, y_pred))

    exit()

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