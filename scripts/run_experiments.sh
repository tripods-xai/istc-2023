# python benchmark_sgrand.py --snr 5.0 --block_size 25 --batch_size 1000 --num_batches 100
# python benchmark_sgrand.py --snr 4.0 --block_size 25 --batch_size 1000 --num_batches 100
# python benchmark_sgrand.py --snr 3.0 --block_size 25 --batch_size 1000 --num_batches 100

# python benchmark_sgrand.py --snr 5.0 --block_size 50 --batch_size 1000 --num_batches 100
# python benchmark_sgrand.py --snr 4.0 --block_size 50 --batch_size 1000 --num_batches 100

# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 5 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 8 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 11 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 16 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 22 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 32 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 45 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 64 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 90 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 128 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 181 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 256 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 362 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 512 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 724 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 1024 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 2048 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs
# python estimate_convcode_conditional_entropy.py --encoder conv_13_7_10 --snr 6.0 --block_len 4096 --batch_size 10000 --num_samples 10000 --no_cuda --logdir ../logs

# python run_experiment.py --experiment_id train_turbo_table_fourier_batch_random --log_every 20
# python run_experiment.py --experiment_id train_turbo_table_batch_random --log_every 20

# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_6 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_5 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_4 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_3 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_2 --log_every 5 --no_cuda

# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_8 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_jtree_nonrecursive_7 --log_every 5 --no_cuda

# python run_experiment.py --experiment_id estimate_ce_turboae_binary_jtree_nonrecursive_4 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turboae_binary_jtree_nonrecursive_1 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turboae_binary_jtree_nonrecursive_2 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turboae_binary_jtree_nonrecursive_3 --log_every 5 --no_cuda

# python run_experiment.py --experiment_id estimate_ce_turbo_random_jtree_nonrecursive_4 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_random_jtree_nonrecursive_1 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_random_jtree_nonrecursive_2 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_random_jtree_nonrecursive_3 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_tae_approx_jtree_nonrecursive_4 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_tae_approx_jtree_nonrecursive_1 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_tae_approx_jtree_nonrecursive_2 --log_every 5 --no_cuda
# python run_experiment.py --experiment_id estimate_ce_turbo_tae_approx_jtree_nonrecursive_3 --log_every 5 --no_cuda

# python run_experiment.py --experiment_id estimate_xe_turbo_random_nonsys_bcjr_varying_block_len --log_every 10
# python run_experiment.py --experiment_id estimate_xe_turbo_tae_approx_nonsys_bcjr_varying_block_len --log_every 10
# python run_experiment.py --experiment_id estimate_xe_conv_random_nonsys_bcjr_varying_block_len --log_every 10
# python run_experiment.py --experiment_id estimate_xe_conv_tae_approx_nonsys_bcjr_varying_block_len --log_every 10

# python run_experiment.py --experiment_id retrain_original_turboae_binary_block_len_40 --log_every 500
# python run_experiment.py --experiment_id train_turboae_w9_first_no_front_small_batch_block_len_40_2 --log_every 500

python run_experiment.py --experiment_id benchmark_turboae_40_2_jtree --log_every 10
python run_experiment.py --experiment_id benchmark_turboae_binary_finetuned_jtree --log_every 10
python run_experiment.py --experiment_id benchmark_turboae_cont_finetuned_jtree --log_every 10

