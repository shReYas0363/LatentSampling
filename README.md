## Abstract

Deep generative models are widely assumed to synthesize unseen combinations of features by mixing individual factors learned during training. Yet even architectures explicitly optimized for disentanglement frequently fall short at this kind of zero-shot compositional generalization. This repository introduces a sampling driven diagnostic framework designed to test whether a pretrained generative model can be steered into generating held-out feature combinations, and to pinpoint the underlying cause when it fails.

Using a pretrained $\beta$-TCVAE alongside trained attribute classifiers, we construct an unnormalized latent-space posterior by multiplying the model's prior with the classifier likelihoods. Conditional samples are then drawn from this distribution via Rejection Sampling, Random-Walk Metropolis–Hastings, and the Metropolis-Adjusted Langevin Algorithm (evaluated both with and without annealing). We train the models on the dSprites dataset while holding out specific shape-attribute pairings, and assess each model using a two part evaluation protocol:

* **Attribute-only queries:** Shape constraints are omitted to observe whether the held-out shape naturally arises when sampling within the target attribute region.
* **Conjunction queries:** Both the target shape and attribute are explicitly conditioned on to determine whether the model can synthesize the unseen pairing on demand.

Results from the attribute-only evaluations show that while the models easily navigate to the desired attribute region, achieving scale and position likelihoods on par with fully trained baselines. The generated outputs are overwhelmingly dominated by shapes encountered during training. Under explicit conjunction queries, the samplers either yield heavily distorted generations that miss the target geometry or produce incorrect shapes that still manage to achieve high classifier scores. Because these outcomes hold across four distinct sampling algorithms and remain consistent over fifteen independent runs per method, the bottleneck stems from the learned representation rather than the inference technique.

Passing ground truth held-out images through the encoder decoder pipeline further reveals that while the decoder preserves scale and position, it renders an imperfect, degraded approximation of the target shape. Even though the resulting geometry remains recognizable to the human eye, this minor degradation causes classifier confidence to drop, which suppresses the posterior density in those latent regions and leads the samplers to reject them. Consequently, this breakdown cannot be traced to the decoder or the prior alone, but instead emerges from how they interact, demonstrating that the disentangled VAE examined here fails to achieve compositional generalization.

# dSprites — Beta-TCVAE Sampling Experiments

> **Dataset Download:** The `dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz` dataset is not included in this repository. Download it from the official Google DeepMind repository:
> - Repository: https://github.com/google-deepmind/dsprites-dataset
> - Direct file: https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
> Place the downloaded `.npz` file in the root of this project directory before running training or evaluation scripts.

# Set Up the Project with `uv`

**1. Download the ZIP file** and extract it.
**2. Navigate into the extracted project directory**
```bash
cd <project-directory>
```
**3. Create the virtual environment**

```bash
uv venv
```
**4. Activate the virtual environment**

```bash
source .venv/bin/activate
```
**5. Install dependencies**

```bash
uv sync
```


## Training

`train_vae.py` — Train Beta-TCVAE
```
python train_vae.py --latent-dim 10 --beta 4 --epochs 15 --checkpoint-path FinalMods/model.pth
```
**Note:** Add the left out conditions manually in the DSprites Dataset Class under zero_shot_mask before running this script. 

`classifier.py` — Train attribute classifiers (shape/scale/position), 80-10-10 split, auto test
```
python classifier.py --model-type all --checkpoint-dir Classifiers --epochs 5
```

`train_gmm_proposal.py` — Fit GMM proposal to the aggregate posterior

```
python train_gmm_proposal.py --vae-checkpoint FinalMods/model.pth
```
*GMM on the aggregate posterior was also used on the samplers, but that may not show their true capabilities at all, as they might just overfit the plausible latent codes of seen combinations*


## Sampling

`rejection_sampler.py` — Rejection sampling
```
python rejection_sampler.py --vae-checkpoint FinalMods/model.pth --attributes shape,scale --target-classes 2,1 --classifier-checkpoints Classifiers/shape.pth,Classifiers/scale_fine.pth --n-accept-target 500
```
*ESS was computed using Importance Sampling weights of all proposed samples parallelly. Included in code, but not utilized or reported in any way.*

`mh_sampler.py` — Random-Walk Metropolis-Hastings
```
python mh_sampler.py --vae-checkpoint FinalMods/model.pth --attributes shape --target-classes 0 --classifier-checkpoints Classifiers/shape.pth --n-chains 100 --n-steps 6000
```

`mala_sampler.py` — MALA (with/without annealing)
```
python mala_sampler.py --vae-checkpoint FinalMods/model.pth --attributes shape --target-classes 0 --classifier-checkpoints Classifiers/shape.pth --anneal-temperature
```

`sampling_common.py` — Shared utilities for all samplers
```
# Checkpoints

`FinalMods/` — Trained VAE checkpoints

`Classifiers/` — shape.pth, scale_fine.pth, position.pth, position_y.pth
```

## Example Experiment Scripts for Sampling 

`run_experiments_2a_MidSq.sh` — All 4 samplers for medium-scale square combos

`run_experiments_3a_LeftMidScaleHe.sh` — All 4 samplers for left-mid-scale heart combos

## Visualization

`visualize_zero_shot_reconstruction.py` — Side-by-side original vs VAE reconstruction of zero-shot samples
```
python visualize_zero_shot_reconstruction.py --checkpoint-path FinalMods/model.pth --num-samples 16
```
**Note:** Add the left out conditions manually in the DSprites Dataset Class under zero_shot_mask before running this script.

`visualize_shape_histogram.py` — Shape-class histogram per sampler from last-trial samples - uses the results directory of the samplers
```
python visualize_shape_histogram.py --results-root ./results_2a_MidSquare --combo-name 2_square_mid --vae-checkpoint FinalMods/model.pth --latent-dim 10 --title "Shape Distribution"
```

`visualize_methods_comparison.py` — n random decoded samples from the last trial of each sampler side by side
```
python visualize_methods_comparison.py --results-root ./results_2a_MidSquare --combo-name 2_square_mid --n-samples 25
```

## Analysis

`analyse_latent.py` — Latent space analysis, traversals and disentanglement metrics.

`LatentAnalysis.ipynb` — Notebook for latent dimensions and disentanglement exploration

## Checkpoints

`FinalMods/` — Trained VAE checkpoints

`Classifiers/` — shape.pth, scale_fine.pth, position.pth, position_y.pth

## Misc

`gmm.py`, `gmm_save_load.py` — GMM model and checkpoint helpers

`utils.py` — Small shared utilities for GMM 

