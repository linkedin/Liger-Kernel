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

## Adding a New Kernel

1. **Create Your Kernel**
Add your kernel implementation in src/liger_kernel/.

2. **Add Unit Tests**
Create unit tests for your kernel in the tests directory. Ensure that your tests cover all kernel functionalities.

## Run test
1. **Install the package**
   ```sh
   pip install -e .[dev]
   ```
2. **Execute test**
   ```sh
   make test
   ```

## Submit PR
Submit the PR followed by the PR template.
