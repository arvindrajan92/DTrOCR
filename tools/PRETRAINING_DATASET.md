# Pre-training dataset

Although it is feasible to prepare DTrOCR's pre-training dataset through the execution of a singular script, it is advisable and more efficient to proceed by visiting pertinent websites in order to download the necessary scripts and data, adhering to the following outlined steps.

## Download font files

Download fonts from [Google Fonts](https://github.com/google/fonts).

## Download text data

Download [The Pile](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated) dataset from Hugging Face to a local folder. Note that the dataset encompasses 451 GB of data; thus, it is imperative to verify the availability of sufficient storage capacity on your drive prior to download.

## Generate printed texts

Produce printed and "handwritten" word images and their corresponding texts using the [Text Renderer](https://github.com/oh-my-ocr/text_renderer).

## Generate handwritten texts

Produce equal amount of synthetic handwritten word images and their corresponding texts using [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator).

## Reorganise data to desired structure