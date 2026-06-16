"""ICRA benchmark human mixes: e=3 ORCA/SFM policy sweep, e=4 PL/PLD policy sweep, e=5 PL/PLD density sweep."""

# Keys: orca, sf (social force), linear (CV), static, powerlaw, pledestrians
HUMAN_MIX_PRESETS = {
    # e=3 — ORCA / SFM policy mixture sweep
    'sfm_only': {'orca': 0, 'sf': 15, 'linear': 0, 'static': 0, 'powerlaw': 0, 'pledestrians': 0},
    'orca_only': {'orca': 15, 'sf': 0, 'linear': 0, 'static': 0, 'powerlaw': 0, 'pledestrians': 0},
    'mix1': {'orca': 8, 'sf': 7, 'linear': 0, 'static': 0, 'powerlaw': 0, 'pledestrians': 0},
    'mix2': {'orca': 5, 'sf': 5, 'linear': 2, 'static': 3, 'powerlaw': 0, 'pledestrians': 0},
    'mix3': {'orca': 4, 'sf': 4, 'linear': 4, 'static': 3, 'powerlaw': 0, 'pledestrians': 0},
    # e=4 — Power Law / PLEdestrians policy mixture sweep (parallel to e=3)
    'PL_only': {'orca': 0, 'sf': 0, 'linear': 0, 'static': 0, 'powerlaw': 15, 'pledestrians': 0},
    'PLD_only': {'orca': 0, 'sf': 0, 'linear': 0, 'static': 0, 'powerlaw': 0, 'pledestrians': 15},
    'PL_mix1': {'orca': 0, 'sf': 0, 'linear': 0, 'static': 0, 'powerlaw': 8, 'pledestrians': 7},
    'PL_mix2': {'orca': 0, 'sf': 0, 'linear': 2, 'static': 3, 'powerlaw': 5, 'pledestrians': 5},
    'PL_mix3': {'orca': 0, 'sf': 0, 'linear': 4, 'static': 3, 'powerlaw': 4, 'pledestrians': 4},
}

# exp_e / exp_se mapping for benchmark grid
BENCHMARK_MIX_CELLS = {
    (3, 0): 'sfm_only',
    (3, 1): 'orca_only',
    (3, 2): 'mix1',
    (3, 3): 'mix2',
    (3, 4): 'mix3',
    (4, 0): 'PL_only',
    (4, 1): 'PLD_only',
    (4, 2): 'PL_mix1',
    (4, 3): 'PL_mix2',
    (4, 4): 'PL_mix3',
}

# e=5 density sweep — same PL/PLD counts as e=0 ORCA/SFM at each se (use exp_e=5, not human_mix)
BENCHMARK_PL_DENSITY_CELLS = {
    (5, 0): {'powerlaw': 2, 'pledestrians': 3},
    (5, 1): {'powerlaw': 5, 'pledestrians': 5},
    (5, 2): {'powerlaw': 7, 'pledestrians': 8},
    (5, 3): {'powerlaw': 10, 'pledestrians': 10},
    (5, 4): {'powerlaw': 12, 'pledestrians': 13},
    (5, 5): {'powerlaw': 15, 'pledestrians': 15},
    (5, 6): {'powerlaw': 17, 'pledestrians': 18},
}

PAPER_MIX_ORDER = [
    'sfm_only', 'orca_only', 'mix1', 'mix2', 'mix3',
    'PL_only', 'PLD_only', 'PL_mix1', 'PL_mix2', 'PL_mix3',
]


def mix_description(name):
    m = HUMAN_MIX_PRESETS[name]
    parts = []
    if m['orca']:
        parts.append(f"{m['orca']} ORCA")
    if m['sf']:
        parts.append(f"{m['sf']} SFM")
    if m['powerlaw']:
        parts.append(f"{m['powerlaw']} Power Law")
    if m['pledestrians']:
        parts.append(f"{m['pledestrians']} PLEdestrians")
    if m['linear']:
        parts.append(f"{m['linear']} CV")
    if m['static']:
        parts.append(f"{m['static']} static")
    return ', '.join(parts) if parts else 'empty'
