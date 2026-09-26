"""Fixed-army baseline using only the policy observation and legal-action mask."""
import numpy as np
from env import config as C
from env.siege_env import deployment_action

SIDES = ("north", "east", "south", "west")


class HeuristicCommander:
    def __init__(self, side=None):
        if side is not None and side not in SIDES:
            raise ValueError("Unknown perimeter side")
        self.side = side
        self.reset()

    def reset(self):
        self.purchases = {t: 0 for t in C.TROOP_TYPES}
        self.chosen_side = None
        self.deployed = 0

    def act(self, obs, mask):
        if obs["vector"][6] == 0:
            for i, troop in enumerate(C.TROOP_TYPES):
                if self.purchases[troop] < C.HEURISTIC_ARMY[troop] and mask[i]:
                    self.purchases[troop] += 1
                    return i
            return C.ACTION_FINISH_SHOPPING
        coverage = obs["grid"][C.GRID_LAYER_DEFENCE_COVERAGE]
        if self.chosen_side is None:
            costs = [sum(coverage[tile] for tile in C.DEPLOY_TILES[s*15:(s+1)*15]) for s in range(4)]
            self.chosen_side = SIDES.index(self.side) if self.side else int(np.argmin(costs))
        indices = list(range(self.chosen_side * 15, (self.chosen_side + 1) * 15))
        # Prefer a central, low-coverage position; alternate neighbours to reduce stacking.
        indices.sort(key=lambda i: (coverage[C.DEPLOY_TILES[i]], abs(i % 15 - 7), i))
        tile = indices[self.deployed % min(3, len(indices))]
        for troop in C.HEURISTIC_DEPLOY_ORDER:
            action = deployment_action(troop, tile)
            if mask[action]:
                self.deployed += 1
                return action
        return C.ACTION_WAIT


def evaluate_from_all_sides(base, *, allow_final=False):
    from env.siege_env import SiegeEnv
    from experiments.rollout import play_episode
    scores = []
    for side in SIDES:
        env = SiegeEnv(motifs=C.MOTIFS, allow_final=allow_final)
        scores.append(play_episode(env, HeuristicCommander(side), options={"spec": base})["summary"]["destruction"])
        env.close()
    return max(scores)
