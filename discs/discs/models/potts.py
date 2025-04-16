"""Potts Energy Function."""

from discs.models import abstractmodel
import jax
import jax.numpy as jnp
import ml_collections


class Potts(abstractmodel.AbstractModel):
  """Potts Distribution (2D cyclic ising model with one-hot representation)."""

  def __init__(self, config: ml_collections.ConfigDict):
    self.sample_shape = config.shape
    self.lambdaa = config.lambdaa
    self.num_categories = config.num_categories
    self.shape = self.sample_shape + (self.num_categories,)
    self.mu = config.mu
    self.init_sigma = config.init_sigma

  def inner_or_outter(self, n, shape):
    if (n[0] / shape - 0.5) ** 2 + (n[1] / shape - 0.5) ** 2 < 0.5 / jnp.pi:
      return 1
    else:
      return -1

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
    x0 = jax.random.randint(
        rnd,
        shape=(num_samples,) + self.sample_shape,
        minval=0,
        maxval=self.num_categories,
        dtype=jnp.int32,
    )
    return x0

  def forward(self, params, x):
      params = params['params']
      if len(x.shape) - 1 == len(self.sample_shape):
        x = jax.nn.one_hot(x, self.num_categories)
      
      # Use the full weight arrays.
      w_h = params[0]  # horizontal weights, shape: (height, width, num_categories)
      w_v = params[1]  # vertical weights, shape: (height, width, num_categories)
      
      # Use jnp.roll to implement cyclic boundaries:
      # Shift by -1 for right and down neighbors; +1 for left and up.
      neighbor_right = jnp.roll(x, shift=-1, axis=2)
      neighbor_left  = jnp.roll(x, shift=1, axis=2)
      neighbor_down  = jnp.roll(x, shift=-1, axis=1)
      neighbor_up    = jnp.roll(x, shift=1, axis=1)
      
      # Sum contributions from the neighbors using the appropriate weights.
      loglikelihood = (
            neighbor_right * w_h +
            neighbor_left  * w_h +
            neighbor_down  * w_v +
            neighbor_up    * w_v
      ) / 2.0
      
      # Add the local bias.
      w_b = params[2]
      loglikelihood = loglikelihood + w_b
      
      # Compute the final energy.
      loglike = x * loglikelihood
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
  return Potts(config.model)
