"""Fixed development panels, selected without policy scores or random draws."""
from env import config as C


def development_panel(pool):
    selected = []
    for motif in C.DEV_MOTIFS:
        candidates = sorted((base for base in pool if base.motif == motif), key=lambda base: base.seed)
        count = min(C.DEV_BASES_PER_MOTIF, len(candidates))
        indices = [round(i * (len(candidates) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
        selected.extend(candidates[index] for index in indices if candidates)
    return selected
