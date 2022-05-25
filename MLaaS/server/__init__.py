from .blackbox import BlackBox
from .storage import Storage


class BlackBoxWrapper:
    def __init__(self, args, blackbox, **kwargs):
        self.args = args
        self.blackbox = blackbox

    def __call__(self, x, **kwargs):
        self.blackbox.query(x, **kwargs)