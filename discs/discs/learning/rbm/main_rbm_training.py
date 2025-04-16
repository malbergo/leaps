"""Training RBM."""

from typing import Sequence
import functools
import os
from absl import app
from absl import flags

from discs.common import data_loader
from discs.common import utils
from discs.common import plot
from discs.learning import train
from discs.models import rbm
from discs.samplers.blockgibbs import RBMBlockGibbsSampler
from flax.core.frozen_dict import unfreeze

import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)


_CONFIG = config_flags.DEFINE_config_file('config')

flags.DEFINE_integer('seed', 1, 'seed')
flags.DEFINE_string('save_root', '', 'root folder for results')

FLAGS = flags.FLAGS


def data_preprocess(x):
  """Preprocess an example from tfds."""
  img = tf.cast(x['image'], tf.float32) / 255.0
  noise = tf.random.uniform(shape=img.shape, maxval=1)
  x = img > noise
  return tf.reshape(x, [-1])


class RBMTrainer(train.Trainer):
  """PCD trainer for RBM."""

  def __init__(self, config, model, sampler):
    super().__init__(config)
    self.model = model
    self.sampler = sampler
    sampler_step = functools.partial(self.sampler.step, model=self.model)
    self.sampler_step = jax.pmap(sampler_step, axis_name='shard')
    self.batch_forward = jax.pmap(self.model.forward, axis_name='shard')

  def build_loss_func(self, rng, batch):
    del rng
    x_positive, x_negative = batch
    def loss_func(params):
      ll_positive = self.model.forward(params, x_positive)
      ll_negative = self.model.forward(params, x_negative)
      loss = jnp.mean(ll_negative - ll_positive)
      return loss, {'loss': loss,
                    'll_pos': jnp.mean(ll_positive),
                    'll_neg': jnp.mean(ll_negative)}
    return loss_func

  def plot_batch(self, step, shared_state, local_state):
    del shared_state
    png_name = '%s/chain-%d.png' % (self.config.experiment.fig_folder, step)
    plot.plot_shareded_image(png_name, local_state.samples, 28, 28, 1,
                             rescale=255.0)

  def batch_processing(self, local_state, shared_state, batch_rng_key, batch):
    samples, sampler_state = local_state.samples, local_state.sampler_state
    for micro_step in range(self.config.experiment.pcd_steps):
      batch_rng_key = jax.random.fold_in(batch_rng_key, micro_step)
      batch_rng = utils.shard_prng_key(batch_rng_key)
      samples, sampler_state = self.sampler_step(
          rng=batch_rng, x=samples, model_param=shared_state.params,
          state=sampler_state)
    local_state = utils.SamplerState(
        step=local_state.step + self.config.experiment.pcd_steps,
        samples=samples, sampler_state=sampler_state)
    batch = (batch, samples)
    return batch, local_state


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  config = _CONFIG.value
  config.experiment.save_root = os.path.join(
      FLAGS.save_root, config.experiment.rbm_config)
  logger = utils.setup_logging(config)
  train_dataset = tfds.load(config.experiment.dataset, split='train')
  with config.unlocked():
    config.model.data_mean = train.get_data_mean(train_dataset)
  with open(os.path.join(config.experiment.save_root, 'config.yaml'), 'w') as f:
    f.write(config.to_yaml())

  global_key = jax.random.PRNGKey(FLAGS.seed)
  model = rbm.build_model(config)
  sampler = RBMBlockGibbsSampler(config.sampler)

  trainer = RBMTrainer(config, model, sampler)
  global_key, init_key = jax.random.split(global_key)
  global_state, local_state = trainer.init_states(init_key)

  train_loader = data_loader.prepare_dataloader(
      train_dataset, config=config.experiment, fn_preprocess=data_preprocess,
      drop_remainder=True, repeat=False)
  train_loader = data_loader.numpy_iter(train_loader)
  final_state = trainer.train_loop(
      logger, global_key, global_state, local_state, train_loader,
      fn_plot=trainer.plot_batch)

  results = {}
  learned_params = unfreeze(final_state.params)
  results['params'] = {}
  results['params']['b_h'] = learned_params['b_h'][0]
  results['params']['b_v'] = learned_params['b_v'][0]
  results['params']['w'] = learned_params['w'][0]
  results['params']['data_mean'] = config.model.data_mean
  results['num_visible'] = config.model.num_visible
  results['num_hidden'] = config.model.num_hidden
  results['num_categories'] = config.model.num_categories
  with open(os.path.join(config.experiment.save_root, 'params.pkl'), 'wb') as f:
      pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
  app.run(main)
