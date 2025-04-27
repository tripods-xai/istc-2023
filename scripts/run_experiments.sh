# Training models
## Retrains turboae cont decoder for block length 40
python run_experiment.py --experiment_id retrain_original_turboae_block_len_40_debug --log_every 500
## Retrains turboae binary decoder for block length 40
python run_experiment.py --experiment_id retrain_original_turboae_binary_block_len_40 --log_every 500
## Trains a fresh turboae with w=9. Used for experiments in paper.
python run_experiment.py --experiment_id train_turboae_w9_first_no_front_small_batch_block_len_40_2 --log_every 500

# Compute the decomposition of the models.
## Fresh trained model
python run_experiment.py --experiment_id benchmark_turboae_40_2_jtree --log_every 10
## Turboae binary
python run_experiment.py --experiment_id benchmark_turboae_binary_finetuned_jtree --log_every 10
## Turboae cont
python run_experiment.py --experiment_id benchmark_turboae_cont_finetuned_jtree --log_every 10

