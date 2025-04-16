"""Ising Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Ising(abstractmodel.AbstractModel):
  """Ising Distribution with Cyclic 2D Lattice."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.shape = config.shape
    self.lambdaa = config.lambdaa
    self.init_sigma = config.init_sigma
    self.mu = config.mu

  def make_init_params(self, rnd):
    params = {}
    # connectivity strength
    params_weight_h = -self.lambdaa * jnp.ones(self.shape)
    params_weight_v = -self.lambdaa * jnp.ones(self.shape)

    params_b = (
        2 * jax.random.uniform(rnd, shape=self.shape) - 1
    ) * self.init_sigma
    indices = jnp.indices(self.shape)
    inner_outter = self.mu * jnp.where(
        (indices[0] / self.shape[0] - 0.5) ** 2
        + (indices[1] / self.shape[1] - 0.5) ** 2
        < 0.5 / jnp.pi,
        1,
        -1,
    )
    params_b += inner_outter
    params['params'] = jnp.array([params_weight_h, params_weight_v, params_b])
    return params

  def get_init_samples(self, rnd, num_samples: int):
    x0 = jax.random.bernoulli(
        rnd,
        shape=(num_samples,) + self.shape,
    ).astype(jnp.int32)
    return x0

  def forward(self, params, x):
    # Unpack parameters
    params = params['params']
    # Full weight arrays are used now since all sites interact cyclically.
    w_h = params[0]  # horizontal coupling weight array (shape = self.shape)
    w_v = params[1]  # vertical coupling weight array (shape = self.shape)
    w_b = params[2]  # bias term (shape = self.shape)
    
    # Convert binary spins in x (0,1) to Ising spins (-1, +1)
    spins = 2 * x - 1

    # Use jnp.roll to implement cyclic boundary conditions.
    # Roll by one pixel in each direction to get the neighbors.
    # Here, axis=1 corresponds to the vertical dimension and axis=2 to the horizontal dimension.
    neighbor_up    = jnp.roll(spins, shift=-1, axis=1)  # downward neighbor (wraps from bottom to top)
    neighbor_down  = jnp.roll(spins, shift=1, axis=1)   # upward neighbor (wraps from top to bottom)
    neighbor_right = jnp.roll(spins, shift=-1, axis=2)  # right neighbor (wraps from right to left)
    neighbor_left  = jnp.roll(spins, shift=1, axis=2)   # left neighbor (wraps from left to right)

    # Multiply the contributions with the corresponding weights.
    # Here, we assume that w_v is used for vertical interactions and w_h for horizontal interactions.
    # The sum of contributions is then divided by 2 to account for double counting.
    message = (neighbor_up * w_v +
               neighbor_down * w_v +
               neighbor_right * w_h +
               neighbor_left * w_h) / 2.0

    # Add the local bias term.
    message = message + w_b

    # Compute the log-likelihood per site and then sum up over the lattice.
    loglike = spins * message
    loglike = loglike.reshape(x.shape[0], -1)
    return -jnp.sum(loglike, axis=-1)
      
  def get_value_and_grad(self, params, x):
    x = x.astype(jnp.float32)  # int tensor is not differentiable

    def fun(z):
      loglikelihood = self.forward(params, z)
      return jnp.sum(loglikelihood), loglikelihood

    (_, loglikelihood), grad = jax.value_and_grad(fun, has_aux=True)(x)
    return loglikelihood, grad


def build_model(config):
  return Ising(config.model)
