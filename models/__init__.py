from .RiRUFold import RiRUFold, RiRFold

__all__ = ['RiRUFold', 'RiRFold', 'get_model']


def get_model(name='rirufold', **kwargs):
    stage_num = int(kwargs.get('stage_num', 5))
    if stage_num < 1:
        raise ValueError('stage_num must be positive')
    if name.lower() in ('rirufold', 'rirfold'):
        return RiRUFold(
            stage_num=stage_num,
            slayers=kwargs.get('slayers', 6),
            llayers=kwargs.get('llayers', 3),
            mlayers=kwargs.get('mlayers', 3),
            channel=kwargs.get('channel', 32),
            mode=kwargs.get('mode', 'train'),
        )
    raise ValueError(f'Unknown model: {name}')
