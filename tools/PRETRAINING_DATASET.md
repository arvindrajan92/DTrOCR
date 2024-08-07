# Pre-training dataset

Although it is feasible to prepare DTrOCR's pre-training dataset through the execution of a singular script, it is advisable and more efficient to proceed by visiting pertinent websites in order to download the necessary scripts and data, adhering to the following outlined steps.

## Download font files

Download fonts from [Google Fonts](https://github.com/google/fonts). A ZIP file (over 1GB) can be downloaded from [here](https://github.com/google/fonts?tab=readme-ov-file#download-all-google-fonts).

Once downloaded, you may use `copy_latin_fonts.py` script to copy latin font files into a dedicated folder. For example: 

```shell
python copy_latin_fonts.py --input <path to Google Fonts folder> --output <font files output folder>
```
## Download text data

Download [The Pile](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) dataset from Hugging Face to a local folder. Note that the dataset encompasses 451 GB of data; thus, it is imperative to verify the availability of sufficient storage capacity on your drive prior to download.

Use the following command line interface (CLI) instruction to initiate the download via terminal:

```shell
sudo huggingface-cli download EleutherAI/the_pile_deduplicated --repo-type dataset --local-dir <insert download path here>
```


## Generate printed texts

Produce printed and "handwritten" word images and their corresponding texts using the [Text Renderer](https://github.com/oh-my-ocr/text_renderer).

## Generate handwritten texts

Produce equal amount of synthetic handwritten word images and their corresponding texts using [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator).

## Reorganise data to desired structure