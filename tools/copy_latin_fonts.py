import os
import glob
import tqdm
import shutil
import argparse

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for copying latin fonts from source folder')
    parser.add_argument('--input', type=str, help='Path to Google Fonts folder')
    parser.add_argument('--font_output', type=str, help='Path to copy font files to')
    parser.add_argument('--list_output', type=str, help='Path to write font list to')
    args = parser.parse_args()

    font_files = glob.glob(os.path.join(args.input, '**', '*.ttf'), recursive=True)

    for font_file in tqdm.tqdm(font_files, desc='Copying font files'):
        meta_file = Path(font_file).with_name("METADATA.pb")

        if meta_file.is_file():
            with open(meta_file, 'r') as f:
                meta = f.read()

            if 'subsets: \"latin\"' not in meta:
                continue

            output_file = Path(args.font_output) / Path(font_file).name
            if not output_file.is_file():
                output_file.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(font_file, output_file)

    print("Writing font list")
    with open(os.path.join(args.list_output, 'font_list.txt'), 'w') as file:
        font_files = glob.glob(os.path.join(args.font_output, '*.ttf'))
        file.write('\n'.join([Path(font_file).name for font_file in font_files]))
