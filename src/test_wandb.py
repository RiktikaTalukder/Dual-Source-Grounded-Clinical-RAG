import wandb
import random

wandb.init(
    project="dual-source-rag",
    name="week2_test_run",
    config={
        "stage": "week2_environment_test"
    }
)

for step in range(5):
    wandb.log({
        "dummy_accuracy": random.random(),
        "dummy_loss": random.random()
    })

wandb.finish()

print("wandb test run complete")