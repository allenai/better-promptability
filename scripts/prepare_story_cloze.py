import sys
from pathlib import Path

import datasets


def main(input_dir: str, cache_dir: str):
    input_dir = Path(input_dir)
    for path in input_dir.iterdir():
        path = path.resolve()
        if path.is_dir():
            name = path.name
            print(f"Loading {name}")
            d = datasets.load_dataset(path)
            print(f"Caching {name}")
            d.save_to_disk(Path(cache_dir) / name)


if __name__ == "__main__":
    main(*sys.argv[1:])
