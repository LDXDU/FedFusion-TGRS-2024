# FedFusion: Manifold Driven Federated Learning for Multi-satellite and Multi-modality Fusion
Run using the **make** command. Specifically including:
run_dist_all (all four datasets)

# ========================


SAVE_EXCEL     ?= 1
SAVE_PNG       ?= 1
LOG_EPOCH      ?= 50

DATASETS        ?= trento houston13 muufl augsburg

OUTPUT_ROOT     ?= ./output
LOG_ROOT        ?= ./logs
RUN_SCRIPT      ?= main.py


DEFAULT_DEVICE  ?= cuda
DEFAULT_DATASET ?= trento
WITH_SVD        ?= 0


# ========================


RUN_CMD         ?= python
TORCHRUN_CMD    = torchrun --nproc_per_node=2 --rdzv_endpoint="localhost:30000"

