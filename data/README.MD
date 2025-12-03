# Data setup instructions

This folder contains the datasets used by the model for training and validation.

## Folder struture 📂

The `data/` folder should have a flat structure. Subfolders are not scanned during training, so make sure to placed the files directly in `data/`.

## Dataset Format 💽

- Datasets must be plain `.txt` files. Avoid rich formats such as `.docx`, `.pdf`, or `.xlsx`.
- The files do not need any special formatting, but it is recommended to clean up the text (remove extra whitespace, unnecessary symbols, etc.) to get the best results.

## Validation Data 🔍

- Validation datasets are created automatically during training.
- You only need to add training data to enrich the model.

## Choosing data 🔭

- The data you choose should be aligned with the characteristics and objectives of your model. For example, if your model is meant to generate casual dialogue you can include conversational text, but if it's for literature, try including narrative text.
- For testing, you can use [Project Gutenberg](https://www.gutenberg.org/), which provides free public-domain books.
- Make sure to follow copyright and licensing rules 📜 when downloading or using data.
