import re
import torch


def read_state_dict(path, device='cpu'):
    try:
        state = torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(path), map_location=device)
    if not isinstance(state, dict):
        raise ValueError(f'Checkpoint must contain a state dict: {path}')
    for key in ('state_dict', 'model_state_dict'):
        if key in state:
            state = state[key]
            break
    return {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}


def load_checkpoint(model, path, device='cpu'):
    state = read_state_dict(path, device)
    stages = {
        int(match.group(1))
        for key in state
        for match in [re.match(r'decos\.(\d+)\.', key)]
        if match
    }
    if stages and stages != set(range(model.stage_num)):
        raise ValueError(
            f'Checkpoint has {max(stages) + 1} stages, but model was initialized with '
            f'stage_num={model.stage_num}. Set --admm-stage-num / --stage-num accordingly.'
        )
    model.load_state_dict(state, strict=True)
    return model
