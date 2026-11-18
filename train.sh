
python visualize_methods_comparison.py  --results-root /home/gandalf/Thesis_results/results_2a_MidSquare     --combo-name 1_any_mid     --vae-checkpoint /home/gandalf/objects/FinalMods/betatcvae_dim10_beta4_MidSq.pth     --latent-dim 10     --title "Medium Scale Images" --n-samples 16
python visualize_methods_comparison.py  --results-root /home/gandalf/Thesis_results/results3     --combo-name 2_square_mid     --vae-checkpoint /home/gandalf/objects/FinalMods/betatcvae_dim10_beta4_MidSq.pth     --latent-dim 10     --title "Medium Scale Squares" --n-samples 16
python visualize_shape_histogram.py  --results-root /home/gandalf/Thesis_results/results_2a_MidSquare     --combo-name 1_any_mid     --vae-checkpoint /home/gandalf/objects/FinalMods/betatcvae_dim10_beta4_MidSq.pth     --latent-dim 10     --title "Medium Scale Images" 

