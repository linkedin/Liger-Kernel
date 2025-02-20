.PHONY: test checkstyle test-convergence all serve build clean


all: checkstyle test test-convergence

# Command to run pytest for correctness tests
test:
	python -m pytest --disable-warnings test/ --ignore=test/convergence

# Command to run ruff for linting and formatting code
checkstyle:
	ruff check .; ruff_check_status=$$?; \
	ruff format --check .; ruff_format_status=$$?; \
	ruff check . --fix; \
	ruff format .; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
		exit 1; \
	fi

# Command to run pytest for convergence tests
# We have to explicitly set HF_DATASETS_OFFLINE=1, or dataset will silently try to send metrics and timeout (80s) https://github.com/huggingface/datasets/blob/37a603679f451826cfafd8aae00738b01dcb9d58/src/datasets/load.py#L286
test-convergence:
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/fp32/test_mini_models.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/fp32/test_mini_models_multimodal.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/fp32/test_mini_models_with_logits.py

	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/bf16/test_mini_models.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/bf16/test_mini_models_multimodal.py
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings test/convergence/bf16/test_mini_models_with_logits.py

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

# MkDocs Configuration
MKDOCS = mkdocs
CONFIG_FILE = mkdocs.yml

# MkDocs targets
serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

build:
	$(MKDOCS) build -f $(CONFIG_FILE)

clean:
	rm -rf site/
