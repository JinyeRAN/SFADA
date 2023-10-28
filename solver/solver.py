solvers = {}
def register_solver(name):
    def decorator(cls):
        solvers[name] = cls
        return cls
    return decorator

def get_solver(name, *args, kwargs={}):
    solver = solvers[name](*args, **kwargs)
    return solver

class BaseSolver:
    """
    Base DA solver class
    """
    def __init__(self, net, ema, trained_generator, src_proto, tgt_memorybank,
                 src_loader, tgt_loader, tgt_sup_loader,
                 tgt_unsup_loader, joint_sup_loader, tgt_opt, ada_stage, device, cfg):
        self.net = net
        self.ema = ema
        self.generator = trained_generator
        self.src_prototype = src_proto
        self.tgt_memorybank = tgt_memorybank
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.tgt_sup_loader = tgt_sup_loader
        self.tgt_unsup_loader = tgt_unsup_loader
        self.joint_sup_loader = joint_sup_loader
        self.tgt_opt = tgt_opt
        self.ada_stage = ada_stage
        self.device = device
        self.cfg = cfg

    def solve(self, epoch):
        pass