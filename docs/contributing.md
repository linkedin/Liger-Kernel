

Thank you for your interest in contributing to Liger-Kernel! This guide will help you set up your development environment, add a new kernel, run tests, and submit a pull request (PR).

!!! Note
    ### Maintainers
    @ByronHsu(admin) @qingquansong @yundai424 @kvignesh1420 @lancerts @JasonZhu1313 @shimizust

## Interested in the ticket?

Leave `#take` in the comment and tag the maintainer. 

## Setting Up Your Development Environment

!!! Note
     1. **Clone the Repository**
     ```sh
    git clone https://github.com/linkedin/Liger-Kernel.git
    cd Liger-Kernel
      ```
     2. **Install Dependencies and Editable Package**
     ```
     pip install . -e[dev]
     ```
     If encounter error `no matches found: .[dev]`, please use
     ```
     pip install -e .'[dev]'
     ```

## Structure

!!! Info
    ### Source Code

    - `ops/`: Core Triton operations.
    - `transformers/`: PyTorch `nn.Module` implementations built on Triton operations, compliant with the `transformers` API.

    ### Tests

    - `transformers/`: Correctness tests for the Triton-based layers.
    - `convergence/`: Patches Hugging Face models with all kernels, runs multiple iterations, and compares weights, logits, and loss layer-by-layer.

    ### Benchmark

    - `benchmark/`: Execution time and memory benchmarks compared to Hugging Face layers.

## Adding support for a new model
To get familiar with the folder structure, please refer [here](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#structure.).

#### 1 Figure out the kernels that can be monkey-patched

a) Check the `src/liger_kernel/ops` directory to find the kernels that can be monkey-patched.

b) Kernels like Fused Linear Cross Entropy require a custom lce_forward function to allow monkey-patching. For adding kernels requiring a similar approach, ensure that you create the corresponding forward function in the `src/liger_kernel/transformers/model` directory.

#### 2 Monkey-patch the HuggingFace model

a) Add the monkey-patching code in the `src/liger_kernel/transformers/monkey_patch.py` file.

b) Ensure that the monkey-patching function is added to the `__init__.py` file in the `src/liger_kernel/transformers/` directory.

#### 3 Add Unit Tests

a) Create unit tests and convergence tests for the monkey-patched model in the tests directory. Ensure that your tests cover all functionalities of the monkey-patched model.

## Adding a New Kernel
To get familiar with the folder structure, please refer [here](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#structure.).

1. **Create Your Kernel**
Add your kernel implementation in `src/liger_kernel/`.

2. **Add Unit Tests**
Create unit tests and convergence tests for your kernel in the tests directory. Ensure that your tests cover all kernel functionalities.

3. **Add Benchmark Script**
Add a benchmarking script under `benchmark/scripts` using the naming convention `benchmark_{kernel_name}.py` showing the performance difference between the Liger kernel and HuggingFace.

## Run tests

### Use Makefile to run full tests
1. Run `make test` to ensure correctness.
2. Run `make checkstyle` to ensure code style.
3. Run `make test-convergence` to ensure convergence.

### Run pytest on single file
`python -m pytest test_sample.py::test_function_name`

## Run kernel benchmarks
The `/benchmark` directory contains benchmarking scripts for the individual kernels, demonstrating differences in speed and memory usage between using Liger and HuggingFace module implementations.

1. Run `make run-benchmarks` to run all benchmarking scripts and append data to `benchmark/data/all_benchmark_data.csv`.
   - Existing entries that are the same (based on `kernel_name`, `kernel_provider`, `kernel_operation_mode`, `metric_name`, `x_name`, `x_value`, `extra_benchmark_config_str`, and `gpu_name`) will not be overwritten.
2. Run `make run-benchmarks OVERWRITE=1` to overwrite any existing entries that have the same configuration.
3. Run `python benchmark/scripts/benchmark_{kernel_name}.py` to run an individual benchmark.
4. You can use the `benchmark/benchmarks_visualizer.py` script to generate visualizations from the CSV, these are then saved to the `benchmark/visualizations` directory (note: this directory is not tracked by git).

## Submit PR
Fork the repo, copy and paste the successful test logs in the PR and submit the PR followed by the PR template (**[example PR](https://github.com/linkedin/Liger-Kernel/pull/21)**).

!!! Warning "Notice"
    As a contributor, you represent that the code you submit is your original work or that of your employer (in which case you represent you have the right to bind your employer).  
    By submitting code, you (and, if applicable, your employer) are licensing the submitted code to LinkedIn and the open source community subject to the BSD 2-Clause license.

#### Release (Maintainer only)

1. Bump the version in pyproject.toml to the desired version (for example, `0.2.0`)
2. Submit a PR and merge
3. Create a new release based on the current HEAD, tag name using `v<version number>` for example `v0.2.0`. Alternatively, If you want to create release based on a different commit hash, `git tag v0.2.0 <commit hash> && git push origin v0.2.0`, and create release based on this tag
4. Adding release note: Minimum requirement is to click the `Generate Release Notes` button that will automatically generates 1) changes included, 2) new contributors. It's good to add sections on top to highlight the important changes.
5. New pip uploading will be triggered upon a new release. NOTE: Both pre-release and official release will trigger the workflow to build wheel and publish to pypi, so please be sure that step 1-3 are followed correctly!

!!! Note "Notes on version"
      Here we follow the [sematic versioning](https://semver.org/). Denote the version as `major.minor.patch`, we increment:

      - Major version when there is backward incompatible change.
      - Minor version when there is new backward-compatible functionality.
      - Patch version for bug fixes.