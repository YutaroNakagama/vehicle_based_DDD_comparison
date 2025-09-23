"""Variational Autoencoder (VAE)-based feature augmentation for domain generalization.

This module trains a lightweight VAE on numeric features and generates synthetic
samples from the latent space to improve model robustness against domain shifts.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf


def sampling(args):
    """
    Reparameterization trick for sampling from a Gaussian distribution.

    Parameters
    ----------
    args : tuple of (tensor, tensor)
        Tuple containing:
        - ``z_mean`` : Mean tensor of shape (batch, latent_dim).
        - ``z_log_var`` : Log-variance tensor of shape (batch, latent_dim).

    Returns
    -------
    tensor
        Sampled latent vector of shape (batch, latent_dim).
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_vae(input_dim: int, latent_dim: int = 10):
    """
    Construct a Variational Autoencoder (VAE) with encoder and decoder networks.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    latent_dim : int, default=10
        Size of the latent space.

    Returns
    -------
    tuple
        A tuple ``(vae, encoder, decoder)`` where:
        - ``vae`` : keras.Model  
            Full VAE model with custom loss.  
        - ``encoder`` : keras.Model  
            Encoder mapping inputs to latent mean, log-variance, and sampled z.  
        - ``decoder`` : keras.Model  
            Decoder mapping latent vectors back to feature space.
    """
    # Encoder
    inputs = Input(shape=(input_dim,), name='encoder_input')
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(32, activation='relu')(latent_inputs)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(input_dim, activation='linear')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    # VAE Loss
    reconstruction_loss = mse(inputs, outputs) * input_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1) * -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam())

    return vae, encoder, decoder


def vae_augmentation(X: pd.DataFrame, augment_ratio: float = 0.3,
                     epochs: int = 30, batch_size: int = 32) -> pd.DataFrame:
    """
    Generate augmented samples using a Variational Autoencoder (VAE).

    Parameters
    ----------
    X : pandas.DataFrame
        Original dataset containing numeric features.
    augment_ratio : float, default=0.3
        Fraction of synthetic samples to generate relative to ``len(X)``.
    epochs : int, default=30
        Number of training epochs for the VAE.
    batch_size : int, default=32
        Batch size used during VAE training.

    Returns
    -------
    pandas.DataFrame
        Concatenated DataFrame of original and VAE-generated samples.
    """
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

