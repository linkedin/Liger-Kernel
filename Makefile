.PHONY: test checkstyle test-convergence all


all: checkstyle test test-convergence

# Command to run pytest for correctness tests
test:
	python -m pytest --disable-warnings test/ --ignore=test/convergence

# Command to run flake8 (code style check), isort (import ordering), and black (code formatting)
# Subsequent commands still run if the previous fails, but return failure at the end
checkstyle:
	ruff check . --fix; ruff_check_status=$$?; \
	ruff format .; ruff_format_status=$$?; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
		exit 1; \
	fi

# Command to run pytest for convergence tests
# We have to explicitly set HF_DATASETS_OFFLINE=1, or dataset will silently try to send metrics and timeout (80s) https://github.com/huggingface/datasets/blob/37a603679f451826cfafd8aae00738b01dcb9d58/src/datasets/load.py#L286
test-convergence:
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/test_mini_models.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/test_mini_models_multimodal.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/test_mini_models_with_logits.py

# Command to run all benchmark scripts and update benchmarking data file
# By default this doesn't overwrite existing data for the same benchmark experiment
# run with `make run-benchmarks OVERWRITE=1` to overwrite existing benchmark data
BENCHMARK_DIR = benchmark/scripts
BENCHMARK_SCRIPTS = $(wildcard $(BENCHMARK_DIR)/benchmark_*.py)
OVERWRITE ?= 0

run-benchmarks:
	@for script in $(BENCHMARK_SCRIPTS); do \
		echo "Running benchmark: $$script"; \
		if [ $(OVERWRITE) -eq 1 ]; then \
			python $$script --overwrite; \
		else \
			python $$script; \
		fi; \
	done
