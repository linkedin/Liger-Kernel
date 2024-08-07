.PHONY: test checkstyle test-convergence

# Command to run pytest for correctness tests
test:
	pytest --disable-warnings -v test/ --ignore=test/convergence
	

# Command to run flake8 (code stylecheck), isort (import ordering) and black (code formatting)
checkstyle:
	flake8 .
	isort .
	black .

# Command to run pytest for convergence tests
# We have to explicitly set HF_DATASETS_OFFLINE=1, or dataset will silently try to send metrics and timeout (80s) https://github.com/huggingface/datasets/blob/37a603679f451826cfafd8aae00738b01dcb9d58/src/datasets/load.py#L286
test-convergence:
	HF_DATASETS_OFFLINE=1 pytest --disable-warnings -v -s test/convergence