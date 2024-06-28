def get_dataset_path(controller, task):
    task_paths = {
        "mujoco-v1": f"data/mujoco/{controller}/{controller}_ResNet-18.safetensors",
        "mujoco-v2": f"data/mujoco-v2/{controller}/{controller}_ResNet-18.safetensors",
        "mujoco-v3": f"data/mujoco-v3/{controller}/{controller}_ResNet-18.safetensors",
    }
    return task_paths.get(task, "Invalid task")
