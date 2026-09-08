.PHONY: test test-verbose checkstyle test-convergence test-cutedsl test-cutile test-cute all serve build clean


all: checkstyle test test-convergence

# Command to run pytest for correctness tests
test:
	python -m pytest --disable-warnings \
		--ignore=test/convergence \
		--ignore=test/cutedsl \
		--ignore=test/cutile \
		--ignore=test/cute \
		test/

# Command to run pytest for CuTe DSL kernel tests
# Requires nvidia-cutlass-dsl: pip install -e '.[cutedsl]'
test-cutedsl:
	LIGER_KERNEL_IMPL=cutedsl python -m pytest --disable-warnings test/cutedsl/

# Command to run pytest for cuTile kernel tests
# Requires cuda-tile: pip install -e '.[cutile]' (or '.[cutile-tileiras]')
test-cutile:
	LIGER_KERNEL_IMPL=cutile python -m pytest --disable-warnings test/cutile/

# Command to run pytest for the cute fused-MoE tests
# The pure-Python registration/import-guard tests always run; the functional
# tests need the separate liger_cute_kernels (lck) wheel and >=2 CUDA devices.
# LIGER_KERNEL_IMPL is intentionally NOT set to 'cute': selecting it would force
# `import liger_kernel.ops` to load the lck wheel and break the always-on tests.
test-cute:
	python -m pytest --disable-warnings test/cute/

# Command to run pytest for correctness tests with coverage reporting and full test-duration breakdown
test-verbose:
	python -m pytest --disable-warnings \
		--cov=src/liger_kernel \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-config=pyproject.toml \
		--durations=0 \
		--ignore=test/convergence \
		--ignore=test/cutedsl \
		--ignore=test/cutile \
		--ignore=test/cute \
		test/

# Command to run coverage report
coverage:
	coverage report -m

# Command to run ruff for linting and formatting code
checkstyle:
	ruff check --output-format=concise .; ruff_check_status=$$?; \
	ruff format --check --diff .; ruff_format_status=$$?; \
	ruff check . --fix; \
	ruff format .; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
		exit 1; \
	fi

# Command to run pytest for convergence tests
# We have to explicitly set HF_DATASETS_OFFLINE=1, or dataset will silently try to send metrics and timeout (80s) https://github.com/huggingface/datasets/blob/37a603679f451826cfafd8aae00738b01dcb9d58/src/datasets/load.py#L286
test-convergence:
	HF_DATASETS_OFFLINE=1 python -m pytest --disable-warnings \
		test/convergence/fp32/test_lfm2_models.py \
		test/convergence/bf16/test_lfm2_models.py \
		test/convergence/fp32/test_mini_models.py \
		test/convergence/fp32/test_mini_models_multimodal.py \
		test/convergence/fp32/test_mini_models_with_logits.py \
		test/convergence/bf16/test_mini_models.py \
		test/convergence/bf16/test_mini_models_multimodal.py \
		test/convergence/bf16/test_mini_models_with_logits.py

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
SITE_DIR = site

# MkDocs targets

# Serve the documentation 
serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

# Build the documentation into the specified site directory
build:
	$(MKDOCS) build -f $(CONFIG_FILE) --site-dir $(SITE_DIR)

# Clean the output directory
clean:
	rm -rf $(SITE_DIR)/
