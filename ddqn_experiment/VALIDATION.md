# Implementation validation

Validated on 12 October 2026 against GitHub commit
`878bc6f4c1e187af992527f86ebfb706bc84ae1a`.

The inspected environment, agents, training, experiment and version2 Python
files matched the recovered handover copy. Existing tracked scientific code,
pool files, dependency declarations, results and README were not modified.

## Runtime used for the final checks

- Python 3.12.14
- PyTorch 2.13.0+cpu
- NumPy 2.5.2
- Gymnasium 1.3.0
- Stable-Baselines3 2.9.0
- SB3 Contrib 2.9.0
- pytest 9.1.1

Tests ran in an isolated Linux environment. This was a CPU test, not a GPU test
or a SageMaker performance measurement. The broader application dependency set
was not installed or rendered as part of these checks.

## Checks completed

`python -m pytest ddqn_experiment/tests -q`: **12 tests passed**.

Coverage includes:

- Double DQN selects with the online network and evaluates with the target network.
- Illegal high-valued actions are excluded, including from learning targets.
- Random exploration samples only legal actions.
- Terminal targets remain finite even with all-false masks and invalid unused values.
- Safety-guard continuation masks leave the live simulator unchanged.
- Float32 replay values and ring position survive saving and loading.
- Exploration schedules do not change when pausing.
- Pauses before, exactly at, and after the curriculum switch resume with identical
  CPU online/target weights, replay content, observations, RNG state and episode
  logs to an uninterrupted 64-transition run.
- Real optimizer updates change the network weights.
- Uncommitted episode log tails are rolled back on resume.
- Changed training settings, corrupted checkpoints and output overwrites are rejected.
- Development evaluation resumes without changing committed results.
- Exported replay files load with the existing project replay loader.
- Final-pool access is rejected.

The CLI preflight accepted **8,118 training layouts** and **156 development
layouts** using the project's existing source/configuration checks.

The CLI smoke command completed **64 transitions**, **16 optimizer updates**
and **3 completed episodes**, with one incomplete placement episode reset at
the planned curriculum boundary. These tiny counts validate execution only;
they are not pilot performance results.

The installer was checked for a clean installation, an identical repeat
installation, and refusal after changing a required source file. Package hashes
were verified after archive extraction.

## Limits

No 262,144-transition pilot, main training cohort or final evaluation was run.
No DDQN superiority, convergence, training duration or full-run stability is
claimed. CUDA resume and browser playback were not exercised. CPU repeatability
was checked only within the recorded runtime; arbitrary hardware or version
changes are not covered. Main-study and final-comparison protocols still need
to be frozen after the pilot is reviewed.
