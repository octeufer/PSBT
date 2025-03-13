import torch

def learn_rate_adaptation(init_lr: float=0.001, num_warmup_steps: int=3000, global_step: int=1):
    global_steps_int = torch.tensor(global_step).to(torch.int32)
    warmup_steps_int = torch.tensor(num_warmup_steps).to(torch.int32)

    global_steps_float = torch.tensor(global_step).to(torch.float32)
    warmup_steps_float = torch.tensor(num_warmup_steps).to(torch.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = (global_steps_int < warmup_steps_int).to(torch.float32)
    learning_rate = is_warmup * warmup_learning_rate
    return learning_rate