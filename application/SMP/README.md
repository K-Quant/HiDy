# SMP Task Experiment

In this part, we implement the RSR model with HiDy to evaluate the effectiveness of HiDy’s knowledge. Codes are modified from https://github.com/Wentao-Xu/HIST. Please refer to original repo for Qlib data download.

## 0. The directory structure of the code
```shell
.
├── data                # Please download and place dataset here 
├── output              # Please download and place the prediction outputs here
├── pretain        # Please download and place pretrain models (.bin) here
├── backtest.ipynb       # Evaluation Results and Backtesting Demo
├── exp        # Run evaluation experiments through pretain models
├── models       # RSR model
├── utils            # dataloader
└── README.md
```

## 1. Download the required files
Please download the data, output, pretain from Google Drive.
https://drive.google.com/drive/folders/1UQ4_OXcBrgx1Jbb7pwkbijR4banXkKrg?usp=drive_link

## 2. Reproduce experiments
Please refer to the ```backtest.ipynb```