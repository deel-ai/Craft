
import jax
import jaxopt
import jax.numpy as jnp
import numbers
import numpy as np
import scipy.sparse as sp
import time
import warnings
from math import sqrt
from typing import NamedTuple, Optional, Callable, Tuple

import tensorflow as tf
from jaxopt import base 
from typing import Optional
from dataclasses import dataclass
from sklearn.decomposition._nmf import _initialize_nmf as sklearn_initialize_nmf
from sklearn.decomposition import NMF as sk_NMF
from jaxopt._src import implicit_diff as idf

from ..utils import jax_l2


def nmf_value_and_gradcam(nmf, W, feature_model, x, concept_index, factoriseur, kkt_sol):
  x = tf.constant(x)
  x = x[None,:,:,:]
  with tf.GradientTape(persistent=True) as tape:  # persistent important !!

    tape.watch(x)
    gcA, A = feature_model(x)
    tape.watch(gcA)

    A = tf.transpose(A)

    # print(jnp.array(A).shape, jnp.array(W).shape)
    nnls_sol = nmf.init_params((jnp.array(A), jnp.array(W)))
    # print(nnls_sol.primal.shape)
    nnls_sol, backward_fun, state = jax.vjp(nmf.run,
                                            nnls_sol, (jnp.array(A), jnp.array(W)),
                                            has_aux=True)
    ua = tf.constant(nnls_sol.primal)
    tape.watch(ua)
    # print(ua.shape)
    #print(f"[{concept_index}] NNLS error={state.error} Iter Num={state.iter_num}")
    
    ui = ua[0,concept_index]

    with tape.stop_recording():  # avoid the tape to record itself
      [tangeant_vector_loss_fn] = tape.gradient([ui], [ua])

      # print(f"tangeant_vector_loss_fn.shape = {tangeant_vector_loss_fn.shape}")

      # Backward in Block n°2 [Jax]
      cotangeant_primal = jnp.array(tangeant_vector_loss_fn)
      cotangeant_dual   = jnp.zeros_like(nnls_sol.dual_ineq)
      cotangeant_vector_nmf = base.KKTSolution(
          primal = cotangeant_primal,
          dual_eq = None, dual_ineq = cotangeant_dual)  # gradient wrt dual variables is zero.
      # the cotangeant vector is now a tangeant vector
      tangeant_vector_kkt_sol, (tangeant_vector_a, tangeant_vector_dict) = backward_fun(cotangeant_vector_nmf)
      del tangeant_vector_kkt_sol  # unused
      del tangeant_vector_dict     # unused but we could compute it if we want

      # print(f"tangeant_vector_nmf.shape = {tangeant_vector_a.shape}")

    # Resume tape.
    # Backward in Block n°1 [Tf]
    cotangeant_vector_features = tf.constant(tangeant_vector_a)  # same shape as Y
    target = tf.reduce_sum(cotangeant_vector_features * A)  # scalar product
    # print(f"target value = {target}")

  tangeant_vector_x = tape.gradient(target, gcA)  # dui / dx
  gc_map = tf.reduce_sum(gcA * tf.reduce_mean(tangeant_vector_x, (1, 2))[:, None, None, :], -1)
  gc_map = tf.image.resize(tf.nn.relu(gc_map)[:, :, :, None], (224, 224), method='bicubic')
  return gc_map, ui


def nmf_gradcam(feature_model, x, factoriseur, kkt_sol):
  pairs = [nmf_value_and_gradcam(feature_model, x, i, factoriseur, kkt_sol) for i in range(factoriseur.rank)]
  grads = [grad for grad, _ in pairs]
  u = tf.stack([ui for _, ui in pairs])
  jacobian = tf.concat(grads, axis=0)
  return jacobian, u
