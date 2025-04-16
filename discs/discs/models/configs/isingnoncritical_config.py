"""Config file for ising model."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(15, 15),
      num_categories=2,
      lambdaa=0.4*0.7,
      external_field_type=1,
      mu=0.0,
      init_sigma=0.0,
      name='ising',
  )
  model_config['save_dir_name'] = 'ising_non_critical'

  return config_dict.ConfigDict(model_config)
