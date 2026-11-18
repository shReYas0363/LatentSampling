#!/bin/bash
set -e

VAE_CKPT="FinalMods/betatcvae_dim10_beta4_MidSq.pth"
SHAPE_CKPT="Classifiers/shape.pth"
SCALE_CKPT="Classifiers/scale_fine.pth"
POSITION_CKPT="Classifiers/position.pth"
POSITION_Y_CKPT="Classifiers/position_y.pth"
LATENT_DIM=10
SAVE_ROOT="./results_2a_MidSquare"
N_STEPS=6000
BURN_IN=2000
N_CHAINS=100
N_TRIALS=15
N_ACCEPT_TARGET=500
MALA_STEP_SIZE=0.005
MH_PROPOSAL_STD=0.752622   
declare -A COMBOS
COMBOS["1_any_mid"]="scale|1|$SCALE_CKPT"
COMBOS["2_square_mid"]="shape,scale|0,1|$SHAPE_CKPT,$SCALE_CKPT"

for combo_name in "${!COMBOS[@]}"; do
  IFS='|' read -r ATTRS TARGETS CKPTS <<< "${COMBOS[$combo_name]}"

  # ---- Rejection sampling ----------------------------------------------
  uv run python rejection_sampler.py \
    --vae-checkpoint "$VAE_CKPT" --latent-dim $LATENT_DIM \
    --attributes "$ATTRS" --target-classes "$TARGETS" \
    --classifier-checkpoints "$CKPTS" \
    --n-accept-target $N_ACCEPT_TARGET --n-trials $N_TRIALS \
    --save-dir "$SAVE_ROOT/rejection/${combo_name}"

  # ---- MH -------------------------------------------------------------
  uv run python mh_sampler.py \
    --vae-checkpoint "$VAE_CKPT" --latent-dim $LATENT_DIM \
    --attributes "$ATTRS" --target-classes "$TARGETS" \
    --classifier-checkpoints "$CKPTS" \
    --order-mode joint \
    --n-chains $N_CHAINS --n-steps $N_STEPS --burn-in $BURN_IN --proposal-std $MH_PROPOSAL_STD \
    --n-trials $N_TRIALS --save-dir "$SAVE_ROOT/mh/${combo_name}"

  # ---- MALA with annealing ---------------------------------------------
  uv run python mala_sampler.py \
    --vae-checkpoint "$VAE_CKPT" --latent-dim $LATENT_DIM \
    --attributes "$ATTRS" --target-classes "$TARGETS" \
    --classifier-checkpoints "$CKPTS" \
    --order-mode joint \
    --n-chains $N_CHAINS --n-steps $N_STEPS --burn-in $BURN_IN --step-size $MALA_STEP_SIZE \
    --anneal-temperature --n-trials $N_TRIALS \
    --save-dir "$SAVE_ROOT/mala_annealed/${combo_name}"

  # ---- MALA without annealing -------------------------------------------
  uv run python mala_sampler.py \
    --vae-checkpoint "$VAE_CKPT" --latent-dim $LATENT_DIM \
    --attributes "$ATTRS" --target-classes "$TARGETS" \
    --classifier-checkpoints "$CKPTS" \
    --order-mode joint \
    --n-chains $N_CHAINS --n-steps $N_STEPS --burn-in $BURN_IN --step-size $MALA_STEP_SIZE \
    --no-anneal-temperature --n-trials $N_TRIALS \
    --save-dir "$SAVE_ROOT/mala_no_anneal/${combo_name}"
Y-Position
done