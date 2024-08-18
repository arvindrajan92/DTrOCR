import os
import glob
import tqdm
import random
import argparse
import multiprocessing as mp

from pathlib import Path
from decimal import DivisionByZero

def split_number(x: int, y: int) -> list[int]:
    if y == 0:
        raise DivisionByZero("Cannot divide by zero")
    base_part = x // y
    remainder = x % y
    parts = []
    for i in range(y):
        if i < remainder:
            parts.append(base_part + 1)
        else:
            parts.append(base_part)
    return parts

def get_words(data: tuple[str|Path, int]) -> list[str]:
    with open(data[0], "r") as f:
        words = f.readlines()
    random_words = random.sample(words, data[1])
    return random_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for build a text file with random words from The Pile')
    parser.add_argument('--input', type=str, help='Path to The Pile text files')
    parser.add_argument('--output', type=str, help='Output text file')
    parser.add_argument('--cpu', type=int, default=1, help='Number of processors')
    parser.add_argument('--num', type=int, help='Number of words')
    args = parser.parse_args()

    text_files = glob.glob(os.path.join(args.input, '**', '*.txt'), recursive=True)
    words_per_file = split_number(args.num, len(text_files))

    with mp.Pool(args.cpu) as pool:
        random_words = list(
            tqdm.tqdm(
                pool.imap(get_words, list(zip(text_files, words_per_file))),
                desc='Getting random words from text files',
                total=len(text_files)
            )
        )

    with open(args.output, "w") as f:
        f.write("".join([w for rw in random_words for w in rw]))