import flyte

acc_env = flyte.TaskEnvironment(
    name="gpu_simple",
    resources=flyte.Resources(gpu=1),
    image=flyte.Image.from_debian_base(registry="localhost:30000").with_pip_packages("torch"),
)


@acc_env.task
async def main() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    flyte.init_from_config()
    r = flyte.run(main)
    print(r.url)
