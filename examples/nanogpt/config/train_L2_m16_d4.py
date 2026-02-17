# L2: Conditioning bottleneck (n=1, m=16, d=4)
# Bottleneck: C(256) -> d(4) -> m^2(256). Tests minimal routing code.
# Per-site overhead: 256*4 + 4*256 = 2K
#
# Usage:
#   python train.py config/train_L2_m16_d4.py --seed=42

out_dir = "out-L2-m16-d4"
wandb_run_name = "L2_m16_d4"
wandb_project = "fine-grained-routing"

dataset = "fineweb10B"

# model
n_layer = 24
n_head = 8
n_embd = 256
block_size = 256
dropout = 0.0
bias = False

# training (1B tokens on 3090 24GB)
batch_size = 16
gradient_accumulation_steps = 8
max_iters = 5000
eval_interval = 500
log_interval = 10
eval_iters = 100
checkpoint_interval = 5000

# optimizer
learning_rate = 6e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# lr schedule
warmup_iters = 500
lr_decay_iters = 30000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# Level 2: n=1, m=16, bottleneck d=4
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 20
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
mhc_residual_identity_mix = False
mhc_residual_alpha = 0.01
routing_granularity = 16
routing_bottleneck_dim = 4
