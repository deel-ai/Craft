import jax
import jaxopt
import jax.numpy as jnp
from dataclasses import dataclass
from typing import NamedTuple, Optional, Callable, Tuple
from jaxopt._src import implicit_diff as idf
import jaxopt.base as base
from sklearn.decomposition import non_negative_factorization

from jaxopt.base import AutoOrBoolean, KKTSolution

from ..utils import jax_l2


class NNLSState(NamedTuple):
  """Named tuple containing state information.
  Attributes:
    iter_num: iteration number.
    error: error used as stop criterion, deduced from residuals.
    primal_residuals: relative residuals primal problem.
    dual_residuals: relative residuals dual problem.
    rho: step size in ADMM.
    H_bar: previous value of H_bar, useful for warm start.  
  """
  iter_num: int
  error: float
  primal_residuals: jnp.array
  dual_residuals: jnp.array
  rho: float
  H_bar: jnp.array


class NNLS(base.IterativeSolver):
  """
  Non-Negative Least Square solver for NMF with implicit differentiation
  """

  def __init__(self, max_iter: int = 1000, tolerance: float = 1e-4, 
               conjugate_gradient_tolerance : float = 1e-6, init_mode: str = "bcd", 
               implicit_diff : bool = True, implicit_diff_solver : Optional[Callable] = None, 
               jit : AutoOrBoolean = "auto", unroll : AutoOrBoolean = "auto"):
    self.max_iter = max_iter
    self.tolerance = tolerance
    self.conjugate_gradient_tolerance = conjugate_gradient_tolerance

    assert init_mode in ['zeros', 'bcd']
    self.init_mode = init_mode
    self.implicit_diff = implicit_diff
    self.implicit_diff_solver = implicit_diff_solver

    self.jit = jit
    self.unroll = unroll
    
    self.optimality_function = self.init_optimality_function()
  
  def init_optimality_function(self):
    """
    Instantiate the jaxopt optimality function for Karush–Kuhn–Tucker (KKT) conditions.
    """
    def objective_function(primal_var, params_obj):
      W = primal_var
      Y, U = params_obj
      
      residual = 0.5 * jax_l2(Y - U @ W.T)
      return residual 
    
    def inequality_constraint(primal_var, params_ineq):
      U = primal_var
      # as the inequality are defined such that X <= 0 is imposed
      # and we want positivity, then -X <= 0 mean that X is positive
      return -U

    return idf.make_kkt_optimality_fun(obj_fun=objective_function, 
                                       eq_fun=None, 
                                       ineq_fun=inequality_constraint)
  
  def init_params(self,
                  params_obj: Tuple[jnp.array, jnp.array]):
    """
    Initialize the U,W and prepare the Karush–Kuhn–Tucker (KKT) solution state.
    """
    Y, W = params_obj
    # as mentioned in the paper, `r` is the lowering dimension (number of concept)
    # and `n` the number of points.
    n, r = Y.shape[1], W.shape[1]

    if self.init == 'zeros':
      U = jnp.zeros((n, r))
    
    elif self.init == 'bcd':
      # use sklearn to find a good initialization point
      start_U, start_W = non_negative_factorization(Y.T, H=W.T, update_H=False)
      # sklearn's W and H are inverted: it is not a bug !
      U = jnp.array(start_U.T)
      print("U shape", U.shape)
      assert U.shape == (n, r)
    
    kkt_solution = KKTSolution(primal=U,
                               dual_eq=None,
                               dual_ineq=U)
    return kkt_solution

  def init_state(self, 
                 init_params,
                 params_obj: Tuple[jnp.array, jnp.array]):
    """
    Initialize the Non-negative Least Square state
    """
    Y, W = params_obj
    r = W.shape[1]
    
    rho = jnp.asarray(jnp.sum(W**2) / r)
    
    state = NNLSState(
        iter_num = jnp.asarray(0, dtype=jnp.int32),
        error = jnp.asarray(jnp.inf),
        primal_residuals = jnp.asarray(jnp.inf),
        dual_residuals = jnp.asarray(jnp.inf),
        rho = rho,
        H_bar = init_params.primal.T,
    )

    return state

  def _compute_H_bar(self, H, G, F, U, rho, H_bar):
    # solution to argmin_H | H - H_bar + U|_2 + delta(H)
    # G is PSD so G + rho * I is PSD => Conjugate Gradient can be used
    def matmul(vec):
      return jnp.dot(G, vec) + rho * vec
    right_member = F + rho * (H + U).T
    H_bar, _ = jax.scipy.sparse.linalg.cg(matmul, right_member,
                                          x0=H_bar, tol=self.conjugate_gradient_tolerance)
    return H_bar
  
  def update(self, params, state, params_obj, params_eq, params_ineq):
    """
    Execute one step and update the state of the Non-negative Least Square.

    with n: number of rows (points),
         p: number of columns (features)
         r: dimension of low rank factorization (number of concepts)

    params
      KKTSolution tuple, with params.primal = H and H of shape (m, k)
    state
      NNLSState object.
    params_obj
      Pair (Y, W), Y of shape (n, m) and W of shape (n, k)
    params_eq
      None, present for signature purposes.
    params_ineq
      None, present for signature purposes.

    Returns:
      pair params.
    """

    Y, W = params_obj
    F = W.T @ Y
    G = W.T @ W  # PSD matrix.
    H, U = params.primal, params.dual_ineq

    # ADMM first inner problem.
    H_bar = self._compute_H_bar(H, G, F, U, state.rho, state.H_bar)

    # ADMM second inner problem.
    H = jax.nn.relu(H_bar.T - U)

    # Gradient ascent step on dual variables.
    U = U + H - H_bar.T

    primal_residuals = jax_l2(H - H_bar.T) / jax_l2(H)
    dual_residuals = jax_l2(H - params.primal) / jax_l2(U)
    error = jnp.maximum(primal_residuals, dual_residuals)

    next_params = base.KKTSolution(primal=H, dual_eq=None, dual_ineq=U)

    next_state = NNLSState(
        iter_num=state.iter_num+1,
        error=error,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        rho=state.rho,
        H_bar=H_bar,
    )

    return base.OptStep(next_params, next_state)

  def run(self,
          params: Optional[base.KKTSolution],
          params_obj: Tuple[jnp.array, jnp.array],
          params_eq: None = None,
          params_ineq: None = None):
    if params is None:
      params = self.init_params(params_obj, params_eq, params_ineq)
    return super().run(params, params_obj, params_eq, params_ineq)
