class Action():
    def __init__(self, config: dict):
        self.action_space = config.get("action_space")

    def step(self, action):
        raise NotImplementedError("The step function must be implemented by the subclass.")