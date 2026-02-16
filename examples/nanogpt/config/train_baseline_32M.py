# L0: Scaling baseline 32M -- standard transformer, no HyperConnections
# C=256, ff=1024, 24 layers, ~32M params
#
# Usage:
#   python train.py config/train_baseline_32M.py --seed=42

out_dir = "out-baseline-32M"
wandb_run_name = "baseline_32M"
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
batch_size = 256
gradient_accumulation_steps = 1
max_iters = 15000
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
lr_decay_iters = 15000
min_lr = 6e-5

# dtype
dtype = "bfloat16"

# hyper-connections: DISABLED (baseline)
hc_num_streams = 1
hc_num_fracs = 1
hc_disable = True
