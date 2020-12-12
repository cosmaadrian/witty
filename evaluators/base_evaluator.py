import wandb

class BaseEvaluator(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model

    def evaluate(self):
        raise NotImplementedError

    def trainer_evaluate(self):
        raise NotImplementedError
