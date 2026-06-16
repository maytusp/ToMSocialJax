import time

import wandb


run = wandb.init(
    project="xxx",
    name="wandb_system_probe",
    mode="online",
)

for step in range(30):
    wandb.log({"probe_step": step, "probe_value": step * 2}, step=step)
    time.sleep(1)

run.finish()
