# Contributing to Liger-Kernel

Thank you for your interest in contributing to Liger-Kernel! This guide will help you set up your development environment, add a new kernel, run tests, and submit a pull request (PR).

## Maintainer

@ByronHsu(admin) @qingquansong @yundai424 @kvignesh1420 @lancerts @JasonZhu1313

## Interested in the ticket?

Leave `#take` in the comment and tag the maintainer. 

## Setting Up Your Development Environment

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

## Adding a New Kernel
To get familiar with the folder structure, please refer to https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#structure.

1. **Create Your Kernel**
Add your kernel implementation in `src/liger_kernel/`.

3. **Add Unit Tests**
Create unit tests and convergence tests for your kernel in the tests directory. Ensure that your tests cover all kernel functionalities.

## Run tests

### Use Makefile to run full tests
1. run `make test` to ensure correctness.
2. run `make checkstyle` to ensure code style.
3. run `make test-convergence` to ensure convergence.

### Run pytest on single file
`python -m pytest test_sample.py::test_function_name`

## Submit PR
Fork the repo, copy and paste the successful test logs in the PR and submit the PR followed by the PR template (**[example PR](https://github.com/linkedin/Liger-Kernel/pull/21)**).

> As a contributor, you represent that the code you submit is your original work or that of your employer (in which case you represent you have the right to bind your employer).  By submitting code, you (and, if applicable, your employer) are licensing the submitted code to LinkedIn and the open source community subject to the BSD 2-Clause license.

## Release (maintainer only)

1. Bump the version in setup.py to the desired version (for example, `0.2.0`)
2. Submit a PR and merge
3. Create a new release based on the current HEAD, tag name using `v<version number>` for example `v0.2.0`. Alternatively, If you want to create release based on a different commit hash, `git tag v0.2.0 <commit hash> && git push origin v0.2.0`, and create release based on this tag
4. New pip uploading will be triggered upon a new release. NOTE: Both pre-release and official release will trigger the workflow to build wheel and publish to pypi, so please be sure that step 1-3 are followed correctly!

### Notes on version:
Here we follow the [sematic versioning](https://semver.org/). Denote the version as `major.minor.patch`, we increment:
- Major version when there is backward incompatible change
- Minor version when there is new backward-compatible functionaility
- Patch version for bug fixes
