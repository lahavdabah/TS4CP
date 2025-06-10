from pydantic import BaseModel


class TS4CPConfig(BaseModel):
    n_eval: int
    beta: float


class TS4CP:
    def __init__(self, ts4cp_config: TS4CPConfig):
        self.n_eval = ts4cp_config.n_eval
        self.beta = ts4cp_config.beta

    