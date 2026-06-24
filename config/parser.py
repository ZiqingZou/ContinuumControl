"""
    A simple hierarchical config parser/saver.
    Supports nested dicts/lists, loading/saving from YAML files,
    getting/setting values via dot-separated paths, and freezing.

    Example usage:
        cfg = Config.load('config.yaml')
        print(cfg.model.hidden_size)
        print(cfg.get('model.hidden_size'))
        cfg.set('training.batch_size', 64)
        cfg.save('new_config.yaml')
        cfg.freeze()
"""

import yaml
from copy import deepcopy


class Config:
    def __init__(self, d=None, _frozen=False):
        d = d or {}
        self._frozen = _frozen
        for k, v in d.items():
            setattr(self, k, self._wrap(v))

    def _wrap(self, v):
        if isinstance(v, dict):
            return Config(v, _frozen=self._frozen)
        if isinstance(v, list):
            # wrap dict elements in lists for convenience
            return [self._wrap(x) for x in v]
        return v

    def to_dict(self):
        def unwrap(obj):
            if isinstance(obj, Config):
                return {k: unwrap(v) for k, v in obj.__dict__.items() if k != '_frozen'}
            if isinstance(obj, list):
                return [unwrap(x) for x in obj]
            return deepcopy(obj)
        return unwrap(self)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    def get(self, path, default=None):
        """path: 'model.pos.z.mean'"""
        cur = self
        for part in path.split('.'):
            if isinstance(cur, Config) and hasattr(cur, part):
                cur = getattr(cur, part)
            elif isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                print(f"[Config Warning] Path '{path}' not found.")
                return default
        return cur

    def set(self, path, value):
        if self._frozen:
            raise AttributeError("Config is frozen")
        parts = path.split('.')
        cur = self
        for p in parts[:-1]:
            if not hasattr(cur, p) or getattr(cur, p) is None:
                setattr(cur, p, Config({}))
            cur = getattr(cur, p)
            if not isinstance(cur, Config):
                raise TypeError(f"Cannot traverse into non-mapping at {p}")
        last = parts[-1]
        setattr(cur, last, self._wrap(value))

    def freeze(self):
        """Make the whole tree read-only (simple flag)."""
        self._frozen = True
        for k, v in self.__dict__.items():
            if k == '_frozen':
                continue
            if isinstance(v, Config):
                v.freeze()
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, Config):
                        item.freeze()

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    # convenience repr
    def __repr__(self):
        return f"Config({self.to_dict()})"

	# Optional: catch missing attribute access
    # def __getattr__(self, name):
    #     print(f"[Config Warning] Attribute '{name}' not found.")
    #     return None
