As a contributor, you represent that the code you submit is your original work or that of your employer (in which case you represent you have the right to bind your employer).  By submitting code, you (and, if applicable, your employer) are licensing the submitted code to LinkedIn and the open source community subject to the BSD 2-Clause license. 


# Contributing to Liger-Kernel

Thank you for your interest in contributing to Liger-Kernel! This guide will help you set up your development environment, add a new kernel, run tests, and submit a pull request (PR).

## Setting Up Your Development Environment

1. **Clone the Repository**
   ```sh
   git clone https://github.com/linkedin/Liger-Kernel.git
   cd Liger-Kernel
   ```
2. **Install Dependencies**
3. **Install Liger Package**
   ```
   pip install . -e[dev]
   ```
## Adding a New Kernel
To get familiar with the folder structure, please refer to https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#structure.

1. **Create Your Kernel**
Add your kernel implementation in src/liger_kernel/.

3. **Add Unit Tests**
Create unit tests and convergence tests for your kernel in the tests directory. Ensure that your tests cover all kernel functionalities.

## Run correctness test
1. **Execute test**
run `make test` to ensure correctness.
run `make checkstyle` to ensure code style.
run `make test-convergence` to ensure convergence.

## Submit PR
Fork the repo, copy and paste the successful test logs in the PR and submit the PR followed by the PR template.
