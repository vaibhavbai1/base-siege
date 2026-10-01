"""Display old and V2 episode tables without mixing modes or weighting learners unevenly."""

def evaluation_summary(frame):
    data=frame.rename(columns={'episode_return':'return'}) if 'return' not in frame else frame.copy()
    metrics=[m for m in ('destruction','town_hall','full_clear','return') if m in data]
    if 'condition' not in data or not metrics:
        raise ValueError('Expected an evaluation episode table with condition and outcome columns.')
    groups=['condition']+(['mode'] if 'mode' in data else [])
    if 'learner_seed' in data:
        result=data.groupby(groups+['learner_seed'])[metrics].mean().groupby(groups).mean()
    else:
        result=data.groupby(groups)[metrics].mean()
    return result
