.PHONY: test checkstyle test-convergence

# Command to run pytest for correctness tests
test:
	pytest -m "not convergence" --disable-warnings -v

# Command to run pre-commit checks
checkstyle:
	pre-commit run --all-files -v

# Command to run speed benchmark
benchmark-speed:
	python -m benchmark.launcher speed benchmark/

# Command to run memory benchmark
benchmark-memory:
	python -m benchmark.launcher memory benchmark/

# Command to run pytest for convergence tests
# We have to explicitly set HF_DATASETS_OFFLINE=1, or dataset will silently try to set metrics and timeout (80s) https://github.com/huggingface/datasets/blob/37a603679f451826cfafd8aae00738b01dcb9d58/src/datasets/load.py#L286
test-convergence:
	HF_DATASETS_OFFLINE=1 pytest -m "convergence" --disable-warnings -v -s

###############################################
#   Below commands are for internal use only
###############################################

# Command to run speed benchmark
benchmark-speed-internal:
	python -m benchmark.launcher speed benchmark_internal/

# Command to run memory benchmark
benchmark-memory-internal:
	python -m benchmark.launcher memory benchmark_internal/
