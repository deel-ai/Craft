
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

from jaxopt import base 
from typing import Optional
from dataclasses import dataclass
from sklearn.decomposition._nmf import _initialize_nmf as sklearn_initialize_nmf
from sklearn.decomposition import NMF as sk_NMF
from jaxopt._src import implicit_diff as idf

from .nnls import NNLS, NNLSState
from ..utils import jax_l2

def _make_nmf_kkt_optimality_fun():
  def obj_fun(primal_var, params_obj):
    H1, H2 = primal_var
    Y      = params_obj
    return 0.5 * jax_l2(Y - H1 @ H2.T)

  def ineq_fun(primal_var, params_ineq):
    H1, H2 = primal_var
    return -H1, -H2 # -H1 <= 0 and -H2 <= 0

  return idf.make_kkt_optimality_fun(obj_fun=obj_fun, eq_fun=None, ineq_fun=ineq_fun)


class NMFState(NamedTuple):
  """Named tuple containing state information.

  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals. 
    nnls_state_1: NNLState associated to H1.
    nnls_state_2: NNLState associated to H2.
  """
  iter_num: int
  error: float
  nnls_state_1: NNLSState
  nnls_state_2: NNLSState


@dataclass(eq=False)
class NMF(base.IterativeSolver):
  rank: int
  maxiter: int = 100
  tol: float = 1e-3
  init: Optional[str] = None
  verbose: int = 0
  implicit_diff: bool = True
  implicit_diff_solve: Optional[Callable] = None
  jit: base.AutoOrBoolean = "auto"
  unroll: base.AutoOrBoolean = "auto"
  nnls: NNLS = NNLS()

  def _split_kkt_sol(self, params):
    H1, H2       = params.primal
    U1, U2       = params.dual_ineq
    kkt_sol_1    = base.KKTSolution(primal=H1, dual_eq=None, dual_ineq=U1)
    kkt_sol_2    = base.KKTSolution(primal=H2, dual_eq=None, dual_ineq=U2)
    return kkt_sol_1, kkt_sol_2

  def _merge_kkt_sol(self, kkt_sol_1, kkt_sol_2):
    primal = kkt_sol_1.primal, kkt_sol_2.primal
    dual_ineq = kkt_sol_1.dual_ineq, kkt_sol_2.dual_ineq
    return base.KKTSolution(primal=primal, dual_eq=None, dual_ineq=dual_ineq)

  def init_params(self,
                  params_obj: jnp.array,
                  params_eq: None = None,
                  params_ineq: None = None):
    Y = params_obj
    if self.init == 'bcd':
      sk_nmf = sk_NMF(n_components=self.rank)
      H1 = sk_nmf.fit_transform(Y)
      H2_T = sk_nmf.components_
    else:
      H1, H2_T = sklearn_initialize_nmf(Y, n_components=self.rank, init=self.init)
    H1, H2 = jnp.array(H1), jnp.array(H2_T.T)
    U1, U2 = jnp.zeros_like(H1), jnp.zeros_like(H2)
    params = base.KKTSolution(primal=(H1, H2),
                              dual_eq=None,
                              dual_ineq=(U1, U2))
    return params

  def init_state(self,
                 init_params: base.KKTSolution,
                 params_obj: jnp.array,
                 params_eq: None = None,
                 params_ineq: None = None):
    Y            = params_obj
    kkt_sol_1, kkt_sol_2 = self._split_kkt_sol(init_params)
    nnls_state_1 = self.nnls.init_state(kkt_sol_1, (Y.T, kkt_sol_2.primal))
    nnls_state_2 = self.nnls.init_state(kkt_sol_2, (Y,   kkt_sol_1.primal))
    return NMFState(
        iter_num=jnp.asarray(0, dtype=jnp.int32),
        error=jnp.asarray(jnp.inf),
        nnls_state_1=nnls_state_1,
        nnls_state_2=nnls_state_2,
    )

  def update(self, params, state, params_obj, params_eq, params_ineq):
    """Update state of NMF.
    
    n: number of rows
    m: number of columns
    k: rank of low rank factorization

    Args:
      params: KKTSolution tuple, with params.primal = (H1, H2),
              H1 of shape (n, k), and H2 of shape (m, k)
      state: NMFState object.
      params_obj: Y of shape (n, m).
      params_eq: None, present for signature purposes.
      params_ineq: None, present for signature purposes.

    Returns:
      pair params, 
    """
    H1, H2 = params.primal
    U1, U2 = params.dual_ineq
    Y = params_obj

    kkt_sol_1, kkt_sol_2 = self._split_kkt_sol(params)

    # Solve \|Y.T - H2 H1.T\| = \|Y - H1 H2.T\| for H1
    kkt_sol_1, nnls_state_1 = self.nnls.run(kkt_sol_1, (Y.T, kkt_sol_2.primal))

    # Solve \|Y - H1 H2.T\| for H2
    kkt_sol_2, nnls_state_2 = self.nnls.run(kkt_sol_2, (Y  , kkt_sol_1.primal))

    H1, H2 = kkt_sol_1.primal, kkt_sol_2.primal
    error = jax_l2(Y - H1 @ H2.T) / jax_l2(Y)

    next_params = self._merge_kkt_sol(kkt_sol_1, kkt_sol_2)

    next_state = NMFState(
        iter_num=state.iter_num+1,
        error=error,
        nnls_state_1=nnls_state_1,
        nnls_state_2=nnls_state_2,
    )
    
    return base.OptStep(next_params, next_state)

  def run(self,
          params: Optional[base.KKTSolution],
          params_obj: jnp.array,
          params_eq: None = None,
          params_ineq: None = None):
    if params is None:
      params = self.init_params(params_obj, params_eq, params_ineq)
    return super().run(params, params_obj, params_eq, params_ineq)

  def __post_init__(self):
    self.optimality_fun = _make_nmf_kkt_optimality_fun()
