![Ising Model Animation](figures/ising.gif)

# Official implementation of LEAPS

This is the official implementation of [LEAPS: A discrete neural sampler via locally equivariant neural networks](https://arxiv.org/pdf/2502.10843), a work by [Peter Holderrieth*](https://www.peterholderrieth.com/), [Michael S. Albergo*](http://malbergo.me/), and [Tommi S. Jaakkola](https://people.csail.mit.edu/tommi/).


## Setup environment

```conda env create -f environment.yml```

## Run experiments for Ising and Potts model

1. **Edit configs:** Edit the configs in `ymls/ising_config.yml` or `ymls/potts_config.yml` to account for the filepaths on your server, i.e. change `code_path`, `ckpts_dir` and `results_path`.

2. **Run commands:** Run code with the following commands:

```python main.py --yml=ymls/ising_config.yml```

```python main.py --yml=ymls/potts_config.yml```


<figure align="center">
  <object 
    data="figures/ising_magnetization_and_corr.pdf" 
    type="application/pdf" 
    width="100%" 
    height="600px">
    <p>
      Your browser doesn’t support embedded PDFs.
      <a href="docs/LEAPS_paper.pdf">Download the PDF instead</a>.
    </p>
  </object>
  <figcaption>Figure: The full LEAPS paper in PDF form.</figcaption>
</figure>

## Run experiments for different energy model

To run experiments on a different energy model, you need to define an annealing path $\rho_t$. You can use the examples in `src/modules/rhot.py` for the Ising model and the Potts model as guiding examples.

## Model checkpoints

We provide two model checkpoints for the two Boltzmann distributions we studied:
- Ising model at critical temperature: ```ckpts/ising_final.ckpt```
- Potts model at critical temperature: ```ckpts/potts_final.ckpt```


## Analyis of trained models


## DISCS benchmark

In our work, we benchmarked against MCMC samplers using the [DISCS benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/file/f9ad87c1ebbae8a3555adb31dbcacf44-Paper-Datasets_and_Benchmarks.pdf). To replicate these experiments, perform the following steps:
1. Go to [DISCS repo](https://github.com/google-research/discs) and clone it.
2. Add the model configs in `discs_configs/` that you can find in this repo as a new model config to the DISCS benchmark in `discs/models/configs/` within the DISCS repo.
3. Copy the scripts `discs_configs/isingcritical_run_discs_benchmark.sh` or `discs_configs/pottscritical_run_discs_benchmark.sh` to the DISCS repo.
4. Run the scripts.