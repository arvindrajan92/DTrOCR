import os
import glob
import tqdm
import shutil
import argparse

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for copying latin fonts from source folder')
    parser.add_argument('--input', type=str, help='Path to Google Fonts folder')
    parser.add_argument('--output', type=str, help='Path to copy font files')
    args = parser.parse_args()

    font_files = glob.glob(os.path.join(args.input, '**', '*.ttf'), recursive=True)

    for font_file in tqdm.tqdm(font_files, desc='Copying font files'):
        meta_file = Path(font_file).with_name("METADATA.pb")

        if meta_file.is_file():
            with open(meta_file, 'r') as f:
                meta = f.read()

            if 'subsets: \"latin\"' not in meta:
                continue

            output_file = Path(args.output) / Path(font_file).name
            output_file.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(font_file, output_file)
