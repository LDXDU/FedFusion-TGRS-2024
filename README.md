# FedFusion: Manifold Driven Federated Learning for Multi-satellite and Multi-modality Fusion
Run using the make command. Specifically including:
run_dist_all (all four datasets)

run_dist_augsburg   run_dist_houston13  run_dist_muufl      run_dist_trento
The running results of the four datasets are already available in the output directory.

main.py:
When reasoning, the only_valid (line 208) parameter controls whether predictions are generated only for valid labels.

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

# ========================


.PHONY: all clean \
        run_dist_all

all: run_dist_all

# ========================



run_dist_all: $(addprefix run_dist_,$(DATASETS))

# ========================


define dist_run_template
run_dist_$(1):
	@$(foreach with_svd, 0 1, \
		$(MAKE) single_run \
			DATASET=$(1) \
			RUN_CMD="$(TORCHRUN_CMD)" \
			MODEL_ARCH=Fed_Fusion \
			WITH_SVD=$(with_svd);)
endef
$(foreach ds,$(DATASETS),$(eval $(call dist_run_template,$(ds))))


# ========================


single_run:
	$(RUN_CMD) $(RUN_SCRIPT) \
		--model_arch $(MODEL_ARCH) \
		--dataset_name $(DATASET) \
		--device $(DEFAULT_DEVICE) \
		--with_svd $(WITH_SVD) \
		--output_dir $(OUTPUT_ROOT)/$(DATASET)_$(DEFAULT_DEVICE) \
		--save_excel $(SAVE_EXCEL) \
		--save_png 1 \
		--log_path $(LOG_ROOT)/$(DATASET)_$(MODEL_ARCH)_$(shell date +%Y%m%d_%H%M%S).log \
		--log_epoch 50

# ========================


clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OUTPUT_ROOT)/debug $(LOG_ROOT)/debug.log

print_config:
	@echo "Current Configuration:"
	@echo "  DATASETS:    $(DATASETS)"
	@echo "  MODEL_ARCHS: $(MODEL_ARCHS)"
	@echo "  DEVICES:     $(DEVICES)"
	@echo "  OUTPUT_ROOT: $(OUTPUT_ROOT)"
