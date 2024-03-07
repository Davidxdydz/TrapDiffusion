from create_dataset import presets, models
from argparse import ArgumentParser
from pathlib import Path
from training.datasets import estimate_dataset_size

if __name__ == "__main__":
    all_presets = list(presets.keys())
    parser = ArgumentParser()
    parser.add_argument("--output", "-o", type=str, default="create_datasets.sh")
    parser.add_argument("--safety", type=int, default=3*1024**3)  # 3 GB
    parser.add_argument("--template", default="templates/create_datasets.sh")
    parser.add_argument("--exclude", nargs="*", choices=all_presets + [[]])
    parser.add_argument("presets", nargs="*", choices=all_presets + [[]])

    args = parser.parse_args()
    output_path = Path(args.output)
    template_path = Path(args.template)
    sizes = []
    if args.presets == []:
        args.presets = all_presets
    if args.exclude:
        if args.presets == []:
            args.presets = all_presets
        args.presets = [preset for preset in args.presets if preset not in args.exclude]
    infos = [presets[preset] for preset in args.presets]
    for info in infos:
        samples, size = estimate_dataset_size(
            models[info["model"]],
            info["include_params"],
            info["configs"],
            info["initial_per_config"],
            info["n_timesteps"],
        )
        sizes.append(size + args.safety)
    total_size = sum(sizes)
    text = template_path.read_text()
    text = text.replace("{{num_tasks}}", f"{len(infos)}")
    text = text.replace("{{total_memory}}", f"{total_size/1024**2:.0f}")
    tasks = []
    for size, preset in zip(sizes, args.presets):
        tasks.append(
            f"srun --exclusive --ntasks=1 --mem={size/1024**2:.0f} python create_dataset.py --preset {preset} --quiet &"
        )
    text = text.replace("{{tasks}}", "\n".join(tasks))
    output_path.write_text(text)
