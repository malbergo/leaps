"""Config file for bernoulli potts."""
from ml_collections import config_dict


def get_config():
  model_config = dict(
      shape=(15, 15),
      lambdaa=1.001,
      init_sigma=0.0,
      num_categories=3,
      external_field_type=1,
      mu=0.0,
      name='potts',
  )
  model_config['save_dir_name'] = 'potts_'+str(model_config['num_categories'])

  return config_dict.ConfigDict(model_config)
