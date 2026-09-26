# Source, decisions and limitations

Current design: study revision 2, 8 September 2026. Start with [TEAM_MASTER.md](TEAM_MASTER.md).

## Source boundary

Received file: base_siege_stage5_sync.zip. SHA-256:
650acdfd85271c1fe55c4107d8d31488eedca0410413416ea526757b56b50ece.

The reference copy is unchanged. Vaibhav supplied configuration, entities, structural generation,
A*, the combat scaffold, renderers, HTML theme and 160 original tests. His build order and the
draft A/B documents are available design evidence. The separately referenced v3 plan was absent.

This project adds the Gymnasium interface, simulator fixes, baselines, validated pools, D4
wrapper, features/PPO curriculum, evaluation/protocol code, Streamlit component, notebooks and
recovery tests. PPO is supplied by SB3 Contrib, not claimed as an original algorithm.

## Simulator decisions retained

- Troops cannot enter live walls. They attack and move after wall removal.
- Simultaneous damage is settled before terminal scoring, so last-tick kills count.
- Defence events record splash victims, damage, attacker type and primary impact.
- Final seed protection applies to generation, resets, pool loading and replay loading.
- The current app renders Python snapshots with the supplied palette, layout and glyphs.
  The archived prototype has its own historical engine and is not the current RL environment.

## Study revision 2 decisions

1. Use ten learner seeds per condition by default. A smaller five-seed resource plan must be
   explicit before main training, not selected after seeing the result.
2. Request the actual 262,144-transition budget directly: 32,768 placement and 229,376 joint.
   The old draft's 250,000 became this count after PPO rollout rounding. No recorded old run
   is silently changed or relabelled.
3. Keep the eight-channel, nine-value partial observation and feed-forward policy. A new memory
   architecture or larger observation would create another experimental factor; evaluate the
   current design in the pilot before broadening it.
4. Exclude layered-core from current development performance evaluation. Earlier software checks
   included it, so disclose that history and do not claim a wholly unexamined structure.
5. Expand training diagnostics to up to four spread layouts per development structure and four
   views. Record the exact panel. Use complete eight-view evaluation for comparisons.
6. Add a separate 57,344-transition pilot pair at learner seed 100. The team must review the
   measured evidence before full registration. No automatic quality or convergence claim.
7. Keep complete-update checkpointing, but externalise immutable pool data from worker snapshots.
   Checkpoint schema 3 restores that pool from the verified run inputs.
8. Journal pool candidates and evaluation episodes transactionally. Resume repeats only
   uncommitted units under the same contract; it does not authorise a new hypothesis.
9. Bind final evaluation to one freeze/output and allow compatible recovery. Reserve seeds remain
   unused. A code/model change is not a resumable final run.
10. Keep source archives, draft submissions and old software artifacts. New evidence uses new
    output folders. No S3, cloud training, publishing, commit or push is part of this local revision.

## Baselines and filtering

The heuristic buys 3 tanks, 6 melee, 4 ranged and 3 breakers for 199 resources. It chooses the
lowest-coverage clockwise side segment, prefers central low-coverage tiles, alternates among the
best three positions and deploys breakers, tanks, ranged and melee. Ties are deterministic.

Validation tries all four sides and accepts best-side destruction from 25% through 85%.
The reported heuristic does not receive those four outcomes. The accepted distribution is
uniform over accepted layouts, not guaranteed balanced by structure.

Final testing considers 200 candidate seeds × four structures, then applies that filter.
There may be fewer than 800 accepted layouts. Report candidate and accepted counts. Never fill
rejections using reserve seeds or select a checkpoint using final scores.

## Limits on our claims

D4 transforms the coordinate view of one simulation. It does not establish physically equivariant
combat, where coordinate tie-breaks may differ. Observations are partial and hand-designed.
Combat is deterministic and simplified, without line of sight, collision avoidance or a learning
defender. There is no commercial-game performance claim.

Seed-level intervals are conditional on the chosen evaluation pool, not universal confidence
over every possible task. Secondary comparisons are exploratory. A negative augmentation result
is valid if the comparison is fair.

The frozen final guard is for accidental misuse, not deliberate tampering. Local recovery cannot
protect against deleting the only storage copy. Policy quality and Studio/CUDA recovery remain
unverified until their actual experiments are run.
