"""Prototype manual controls over the unchanged Python combat engine.

Human deployments do not consume time. The trained Gym API still advances ten
battle ticks per decision; this adapter is for interactive demonstrations only.
"""
from env import config as C
from env.combat import deploy_troop, step_tick
from env.siege_env import SiegeEnv


class ManualGame(SiegeEnv):
    def reset(self, **kwargs):
        self.revision = 0
        return super().reset(**kwargs)

    def command(self, kind, *, troop=None, tile=None, count=1):
        s = self.state
        if s.is_finished:
            raise ValueError('The battle has finished. Start a new base.')
        if kind in ('buy', 'refund'):
            if s.phase != 'shopping' or troop not in C.TROOP_TYPES:
                raise ValueError('Buy or refund troops during shopping only.')
            cost = C.TROOP_COSTS[troop]
            if kind == 'buy':
                if s.spent + cost > C.RESOURCE_BUDGET or sum(s.reserve.values()) >= C.MAX_ARMY_SIZE:
                    raise ValueError('Army size or budget limit reached.')
                s.reserve[troop] = s.reserve.get(troop, 0) + 1
                s.spent += cost
            else:
                if not s.reserve.get(troop, 0):
                    raise ValueError('There is no troop of this type to refund.')
                s.reserve[troop] -= 1
                s.spent -= cost
            message = f'{kind.title()} {troop}'
        elif kind == 'start':
            if s.phase != 'shopping' or not any(s.reserve.values()):
                raise ValueError('Buy at least one troop before starting.')
            s.phase = 'battle'
            message = 'Deploy your opening wave, then run the clock.'
        elif kind == 'deploy':
            if (s.phase != 'battle' or troop not in C.TROOP_TYPES or type(tile) is not int
                    or not 0 <= tile < C.N_DEPLOY_TILES or type(count) is not int or count not in (1, 5)):
                raise ValueError('Choose a troop and a green perimeter tile.')
            amount = min(count, s.reserve.get(troop, 0))
            if not amount:
                raise ValueError('No selected troops remain in reserve.')
            for _ in range(amount):
                deploy_troop(s, troop, C.DEPLOY_TILES[tile])
            message = f'{amount} × {troop} deployed at {C.DEPLOY_TILES[tile]}'
        elif kind == 'tick':
            if s.phase != 'battle':
                raise ValueError('Start the battle before running the clock.')
            step_tick(s)
            self.decisions = s.tick // C.TICKS_PER_DECISION
            message = f'Tick {s.tick} · {s.destroyed_fraction:.1%} destroyed'
        else:
            raise ValueError('Unknown manual command.')
        if kind != 'tick':
            s.last_attacks = []
        self.events = [{**a, 'tick': s.tick} for a in s.last_attacks]
        self.revision += 1
        self._record()
        return message

    def snapshot(self):
        frame = super().snapshot()
        frame['revision'] = self.revision
        # Human play has different decision timing; do not present its actions
        # or zero-initialized Gym rewards as a registered policy evaluation.
        for name in ('reward_components', 'reward_totals', 'episode_return'):
            frame.pop(name, None)
        return frame
