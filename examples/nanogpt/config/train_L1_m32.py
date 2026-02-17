# L1: Single H^res routing, no bottleneck (n=1, m=32)
# Residual (256,) reshaped to (32, 8) before H^res.
# Per-site overhead: C * m^2 = 256 * 1024 = 262K (intentionally expensive)
#
# Usage:
#   python train.py config/train_L1_m32.py --seed=42

out_dir = "out-L1-m32"
wandb_run_name = "L1_m32"
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
batch_size = 8
gradient_accumulation_steps = 16
max_iters = 30000
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

# Level 1: n=1 (no bus), mHC enabled for H^res routing only
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = False
mhc = True
sinkhorn_iters = 20
sinkhorn_tau = 0.05
mhc_h_res_proj = "sinkhorn"
mhc_residual_identity_mix = False
mhc_residual_alpha = 0.01
routing_granularity = 32
