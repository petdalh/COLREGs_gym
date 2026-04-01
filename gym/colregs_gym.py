from mchorcrux.numpy_core.gym.mc_gym_csad_numpy import MCGym

from .action import Action

class COLREGsGym(MCGym):
    def __init__(
        self,
        vessel_model,
        dt,
        grid_width,
        grid_height,
        config=None,
        **kwargs,
    ):
        super().__init__(
            dt=dt, grid_width=grid_width, grid_height=grid_height, **kwargs
        )

        self.vessel_action = Action(config)
        self.action_space = self.vessel_action.action_space

    def step(self, action):
        self.vessel_action.step(action)
        return super().step(action)
        
