# damaged_ts_detection
**Master’s Thesis by Can Ersü**

## Repository Overview

This repository contains the implementation of the master's thesis titled **"Automatic Visual Traffic Sign Damage Detection Using Deep Learning Algorithms"**. The project focuses on developing a system for automatically detecting damage to traffic signs using deep learning techniques. The system utilizes YOLO (You Only Look Once) for object detection and an autoencoder for anomaly detection on the German Traffic Sign Detection Benchmark (GTSDB) dataset.

## Table of Contents

- [Repository Overview](#repository-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Thesis Reference](#thesis-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/canersu/damaged_ts_detection.git
    cd damaged_ts_detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

This project uses the GTSDB (German Traffic Sign Detection Benchmark) dataset for traffic sign detection. The dataset consists of images containing traffic signs with varying degrees of damage, captured under different conditions.

- **GTSDB Dataset**: Available [here](http://benchmark.ini.rub.de/?section=gtsdb&subsection=news).

Please ensure you download and preprocess the dataset before running the experiments. Detailed instructions for downloading and preprocessing are provided in the `data_preprocessing.md` file.

## Project Structure

The repository is organized as follows:

- `data/`: Contains the dataset and preprocessed data files.
- `models/`: Contains trained models and their configurations.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model training.
- `src/`: Source code for the project, including scripts for training, evaluation, and utilities.
- `outputs/`: Stores the results of experiments, including logs, metrics, and model checkpoints.
- `README.md`: Project overview and instructions (this file).

## Usage

1. **Training the Model**:
   - To train the YOLO model for traffic sign detection:
     ```bash
     python src/train_yolo.py --config configs/yolo_config.yaml
     ```

   - To train the autoencoder for damage detection:
     ```bash
     python src/train_autoencoder.py --config configs/autoencoder_config.yaml
     ```

2. **Evaluating the Model**:
   - To evaluate the trained models on the test set:
     ```bash
     python src/evaluate.py --config configs/evaluate_config.yaml
     ```

3. **Running Inference**:
   - To run inference on a new set of images:
     ```bash
     python src/inference.py --input data/new_images/ --output results/
     ```

## Results

The results of the experiments, including performance metrics and visualizations, are stored in the `outputs/` directory. Detailed analysis and interpretation of the results are provided in the thesis document.

## Thesis Reference

For more details about the methodology, experiments, and results, please refer to the thesis document:

**Title**: Automatic Visual Traffic Sign Damage Detection Using Deep Learning Algorithms  
**Author**: Can Ersü  
**Institution**: Tallinn University of Technology  
**Year**: 2023

The thesis document is available in the root directory of this repository as `Thesis_Can_Ersu.pdf`.

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and create a pull request with your changes. Ensure that your code follows the established guidelines and is well-documented.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
