# Additional optimization diagnostics

`A2/additional-optimization-diagnostics.tar.gz` and
`A3/additional-optimization-diagnostics.tar.gz` preserve probe scripts, logs,
profiler mappings and kernel duration tables from further optimization trials.
These are diagnostic measurements, separate from the original acceptance
benchmark. Rejected variants remain in these archives.

- Transposed reduction did not compile on the fixed compiler stack.
- Column segmentation, address offsets, hierarchical reduction and compiler
  options did not establish a consistent benefit over the current kernels.
- The A3 compiler-option probe completed its 4096-column capture; its subsequent
  3584-column profiler capture stopped making progress and was terminated. The
  timeout record and process exit status are retained. No compiler option from
  this trial was adopted.
- The first A2 frequency diagnostic completed its token sweep but the independent
  sampler failed to start. The second diagnostic sampled successfully: 28 core
  frequency observations at 800 MHz and 6 at 1800 MHz, with all 34 HBM observations
  at 1600 MHz. Both token sweeps passed 12/12 speed comparisons. This observation
  does not establish that frequency variation caused any earlier failure; these
  partial diagnostic sweeps do not replace two full acceptance rounds.
- The launch-name reference optimization completed full validation in the
  `namecache-port` stage; dimension handling was subsequently moved to the
  native entry in `dimcache-port`. Both stages retain all raw results. The generated launcher and native extension both use
  the installed PyTorch C++ ABI.

No hardware frequency, power limit, benchmark timing, assertion or tolerance was
changed by these probes.

The later LSE, release-compiler and single-buffer probes are preserved in
`A2/dimcache-profiling.tar.gz`. None showed a consistent improvement, so their
compiler options and buffer changes were not adopted. Current implementation
profiling outputs for both platforms are in the respective `dimcache-profiling.tar.gz`
archives; `dimcache-profile-summary.json` summarizes their device kernel durations.

Intermediate raw trial files are packaged in each platform's `optimization-trials.tar.gz`.
The adjacent manifest records the SHA-256 of every archived file. Current
`dimcache-port` and matched-baseline evidence remains directly readable.
