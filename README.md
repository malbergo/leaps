# Official implementation of LEAPS

## Run experiments for Ising and Potts model

1. **Edit configs:** Edit the configs in `ymls/ising_config.yml` or `ymls/potts_config.yml` to account for the filepaths on your server, i.e. change `code_path`, `ckpts_dir` and `results_path`.

2. **Run commands:** Run code with the following commands:

```python main.py --yml=ymls/ising_config.yml```

```python main.py --yml=ymls/potts_config.yml```


## Run experiments for different energy model

To run experiments on a different energy model, you need to define an annealing path $\rho_t$. You can use the examples in `src/modules/rhot.py` for the Ising model and the Potts model as guiding examples.

## Model checkpoints
