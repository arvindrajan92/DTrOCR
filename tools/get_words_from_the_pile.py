import os
import re
import glob
import tqdm
import string
import argparse
import pandas as pd
import multiprocessing as mp

from pathlib import Path


def get_unique_words_from_parquet(data: tuple[str, str]) -> Path:
    parquet_file = Path(data[0])
    output_folder = Path(data[1])

    # read parquet file into dataframe
    df = pd.read_parquet(parquet_file)

    # Step 1: Concatenate all sentences into one string
    concatenated_string = ' '.join(df['text'])

    # Step 2: Remove escape characters
    clean_string = re.sub(r'\s+', ' ', concatenated_string)

    # Step 3: Split the string into words
    words = clean_string.split(' ')

    # Step 4: Get unique words
    unique_words = [w for w in set(words) if w.strip() != '' and all(char in string.printable for char in w)]

    # writing to output folder
    out_file = Path(os.path.join(output_folder,parquet_file.with_suffix('.txt').name))
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with open(out_file, 'w') as file:
        file.write('\n'.join(unique_words))

    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for getting distinct words from The Pile parquet files')
    parser.add_argument('--input', type=str, help='Path to The Pile folder')
    parser.add_argument('--output', type=str, help='Path to write files to')
    parser.add_argument('--cpu', type=int, default=1, help='Number of processors')
    args = parser.parse_args()

    parquet_files = glob.glob(os.path.join(args.input, '**', '*.parquet'), recursive=True)

    with mp.Pool(args.cpu) as pool:
        input_files = list(
            tqdm.tqdm(
                pool.imap(get_unique_words_from_parquet, [(parquet_file, args.output) for parquet_file in parquet_files]),
                desc='Getting distinct words from parquet files',
                total=len(parquet_files)
            )
        )
