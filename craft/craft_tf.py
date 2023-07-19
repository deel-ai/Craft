"""
CRAFT Module for Tensorflow
"""

from abc import ABC, abstractmethod
from typing import Union, Tuple, List, Callable, Dict, Optional, Any

import tensorflow as tf
import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import NotFittedError

from .sobol.sampler import HaltonSequence
from .sobol.estimators import JansenEstimator


class BaseConceptExtractor(ABC):
    """
    Base class for concept extraction models.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.

    """

    def __init__(self, input_to_latent : Callable,
                       latent_to_logit : Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64):

        # sanity checks
        assert(number_of_concepts > 0), "number_of_concepts must be greater than 0"
        assert(batch_size > 0), "batch_size must be greater than 0"
        assert(callable(input_to_latent)), "input_to_latent must be a callable function"

        self.input_to_latent = input_to_latent
        self.latent_to_logit = latent_to_logit
        self.number_of_concepts = number_of_concepts
        self.batch_size = batch_size

    @abstractmethod
    def fit(self, inputs):
        """
        Fit the CAVs to the input data.

        Parameters
        ----------
        inputs : array-like
            The input data to fit the model on.

        Returns
        -------
        tuple
            A tuple containing the input data and the matrices (U, W) that factorize the data.

        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, inputs):
        """
        Transform the input data into a concepts embedding.

        Parameters
        ----------
        inputs : array-like
            The input data to transform.

        Returns
        -------
        array-like
            The transformed embedding of the input data.

        """
        raise NotImplementedError


class Craft(BaseConceptExtractor):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism.

    Parameters
    ----------
    input_to_latent : Callable
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
    latent_to_logit : Callable
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
    number_of_concepts : int
        The number of concepts to extract.
    batch_size : int, optional
        The batch size to use during training and prediction. Default is 64.
    patch_size : int, optional
        The size of the patches to extract from the input data. Default is 64.
    """

    def __init__(self, input_to_latent : Callable,
                       latent_to_logit : Optional[Callable] = None,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64):
        super().__init__(input_to_latent, latent_to_logit, number_of_concepts, batch_size)

        self.patch_size = patch_size
        self.activation_shape = None

    def fit(self, inputs : np.ndarray):
        """
        Fit the Craft model to the input data.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (n_samples, height, width, channels).
            (x1, x2, ..., xn) in the paper.

        Returns
        -------
        (X, U, W)
            A tuple containing the crops (X in the paper),
            the concepts values (U) and the concepts basis (W).
        """
        # extract patches from the input data
        strides = int(self.patch_size * 0.80)
        patches = tf.image.extract_patches(images=inputs,
                                           sizes=[1, self.patch_size, self.patch_size, 1],
                                           strides=[1, strides, strides, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')
        patches = tf.reshape(patches, (-1, self.patch_size, self.patch_size, 3))

        # encode the patches and obtain the activations
        input_width, input_height = inputs.shape[1], inputs.shape[2]
        activations = self.input_to_latent.predict(tf.image.resize(patches, (input_width, input_height), method="bicubic"),
                                                  batch_size=self.batch_size,
                                                  verbose=False)

        assert np.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            activations = tf.reduce_mean(activations, axis=(1, 2))

        # apply NMF to the activations to obtain matrices U and W
        reducer = NMF(n_components=self.number_of_concepts, alpha_W=1e-2)
        U = tf.cast(reducer.fit_transform(tf.nn.relu(activations)), tf.float32)
        W = tf.cast(reducer.components_, tf.float32)

        # store the factorizer and W as attributes of the Craft instance
        self.reducer = reducer
        self.W = tf.cast(W, tf.float32)

        return patches, U, W


    def check_if_fitted(self):
        """Checks if the factorization model has been fitted to input data.

        Raises
        ------
        NotFittedError
            If the factorization model has not been fitted to input data.
        """

        if not hasattr(self, 'reducer'):
            raise NotFittedError("The factorization model has not been fitted to input data yet.")

    def transform(self, inputs : np.ndarray, activations : np.ndarray = None):
        """Transforms the inputs data into its concept representation.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data to be transformed.
        activations: numpy array or Tensor, optional
            Pre-computed activations of the input data. If not provided, the activations
            will be computed using the input_to_latent model.

        Returns
        -------
        U : Tensor
            The concept value (U) of the inputs.
        """
        self.check_if_fitted()

        activations = self.input_to_latent.predict(inputs, batch_size=self.batch_size,
                                                   verbose=False)

        if len(activations.shape) == 4:
            original_shape = activations.shape[:-1]
            activations = np.reshape(activations, (-1, activations.shape[-1]))
        else:
            original_shape = (len(activations),)

        W_dtype = self.reducer.components_.dtype
        U = self.reducer.transform(np.array(activations, dtype=W_dtype))
        U = np.reshape(U, (*original_shape, U.shape[-1]))

        return tf.cast(U, tf.float32)

    def estimate_importance(self, inputs, class_id, nb_design = 32):
        """
        Estimates the importance of each concept for a given class.

        Parameters
        ----------
        inputs : numpy array or Tensor
            The input data to be transformed.
        class_id : int
            The class id to estimate the importance for.
        nb_design : int, optional
            The number of design to use for the importance estimation. Default is 32.

        Returns
        -------
        importances : list
            The Sobol total index (importance score) for each concept.

        """
        self.check_if_fitted()

        U = self.transform(inputs)

        masks = HaltonSequence()(self.number_of_concepts, nb_design = nb_design)
        estimator = JansenEstimator()

        importances = []

        if len(U.shape) == 2:
            # apply the original method of the paper

            for u in U:
                u_perturbated = u[None, :] * masks
                a_perturbated = u_perturbated @ self.W

                y_pred = self.latent_to_logit.predict(a_perturbated, batch_size=self.batch_size,
                                                      verbose=False)
                y_pred = y_pred[:, class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        elif len(U.shape) == 4:
            # apply a re-parameterization trick and use mask on all localization for a given
            # concept id to estimate sobol indices
            for u in U:
                u_perturbated = u[None, :] * masks[:, None, None, :]
                a_perturbated = np.reshape(u_perturbated, (-1, u.shape[-1])) @ self.W
                a_perturbated = np.reshape(a_perturbated, (len(masks), U.shape[1], U.shape[2], -1))

                y_pred = self.latent_to_logit.predict(a_perturbated, batch_size=self.batch_size,
                                                      verbose=False)
                y_pred = y_pred[:, class_id]

                stis = estimator(masks, y_pred, nb_design)

                importances.append(stis)

        return np.mean(importances, 0)
