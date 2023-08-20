from pathlib import Path
from pprint import pprint

dataset_root = Path("../../data/datasets/asl-videos")
print(dataset_root.exists())
index_file = dataset_root / "index.txt"
words_to_scrape = []
with open(index_file, "r") as f:
    for line in f.readlines():
        words_to_scrape.append(line.strip())
pprint(words_to_scrape)