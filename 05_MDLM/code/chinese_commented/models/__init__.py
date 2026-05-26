import importlib

def __getattr__(name):
  if name == 'dit':
    return importlib.import_module('.dit', __package__)
  if name == 'dimamba':
    return importlib.import_module('.dimamba', __package__)
  if name == 'ema':
    return importlib.import_module('.ema', __package__)
  if name == 'autoregressive':
    return importlib.import_module('.autoregressive', __package__)
  raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
