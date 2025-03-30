import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae(input_dim, latent_dim=10):
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(32, activation='relu')(latent_inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(input_dim, activation='linear')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam())

    return vae, encoder, decoder

def vae_augmentation(X, augment_ratio=0.3, epochs=30, batch_size=32):
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_cols].values

    vae, encoder, decoder = build_vae(input_dim=X_numeric.shape[1])
    vae.fit(X_numeric, epochs=epochs, batch_size=batch_size, verbose=0)

    num_augmented_samples = int(len(X_numeric) * augment_ratio)
    z_mean, z_log_var, _ = encoder.predict(X_numeric, verbose=0)

    augmented_data = []
    for _ in range(num_augmented_samples):
        idx = np.random.randint(len(X_numeric))
        epsilon = np.random.normal(size=z_mean.shape[1])
        z_sample = z_mean[idx] + np.exp(0.5 * z_log_var[idx]) * epsilon
        x_decoded = decoder.predict(z_sample.reshape(1, -1), verbose=0)
        augmented_data.append(x_decoded.flatten())

    augmented_df = pd.DataFrame(augmented_data, columns=numeric_cols)
    X_augmented = pd.concat([X.reset_index(drop=True), augmented_df], ignore_index=True)

    return X_augmented

