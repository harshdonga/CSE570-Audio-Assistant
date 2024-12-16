# AI Audio Assistant

## Overview
This project is part of CSE 570 - Wireless and Mobile Network at SBU taken by Prof. Shubham Jain.
It utilizes Python for training and deploying models based on the [FSD50K dataset](https://zenodo.org/records/4060432).
This repository provides tools for training models and running a command-line interface for inference or additional tasks.

## Prerequisites
- Python 3.9 or higher.
- `pip` for managing Python packages.
- Approximately **100GB** of disk space to download and process the FSD50K dataset.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/harshdonga/CSE570-Audio-Assistant
cd CSE570-Audio-Assistant
```

### 2. Create a Virtual Environment
To keep dependencies isolated, it’s recommended to use a virtual environment:

```bash
python3.9 -m venv env
source env/bin/activate
```

### 3. Install Required Dependencies
Install the Python packages required for the project:

```bash
pip install -r requirements.txt
```

### 4. Download the FSD50K Dataset
Follow the instructions provided on the [FSD50K Zenodo page](https://zenodo.org/records/4060432) to download the dataset.

- Ensure the dataset is downloaded into a directory named `fsd50k` within   the root of the project. If using a different directory, update the path in the training script configuration.

### 5. Train the Models
Use `train.py` for training the models

### 6. Run the Command-Line Interface (CLI)
Activate the environment if it’s not already activated:

```bash
source env/bin/activate
```

Run the CLI script:

```bash
python cli.py
```

Follow the prompts in the CLI for further interactions.

## Additional Notes
- **Models**: Models will be saved and read from `models/` directories by default. Ensure these directories exist or modify the script to create them dynamically.
- **Features**: Features & Labels are read from `X/`  & `y/` directories respectively by default, so make sure helper script saves them accordingly

## Acknowledgments
- Dataset: [FSD50K](https://zenodo.org/records/4060432).
- Contributors: Harsh Donga, Drishti Singh.

