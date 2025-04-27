# python run_experiment.py --experiment_id retrain_original_turboae_binary_block_len_40 --log_every 500
python run_experiment.py --experiment_id train_turboae_w9_first_no_front_small_batch_block_len_40_2 --log_every 500

python run_experiment.py --experiment_id benchmark_turboae_40_2_jtree --log_every 10
python run_experiment.py --experiment_id benchmark_turboae_binary_finetuned_jtree --log_every 10
python run_experiment.py --experiment_id benchmark_turboae_cont_finetuned_jtree --log_every 10

