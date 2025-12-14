import yaml

class Config:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.cfg = yaml.safe_load(f)

    def get(self, *keys, default=None):
        d = self.cfg
        for key in keys:
            if key in d:
                d = d[key]
            else:
                return default
        return d