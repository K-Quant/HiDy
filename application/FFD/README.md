
# FFD Task Experiment

In this part, we incorporate HiDy's dynamic knowledge in different hierarchy for FFD.

## 0. The directory structure of the code
```shell
.
├── data              # please download and place dataset here
├── ffd.py       # file to run the experiment
└── README.md
```

## 1. Download and place the data
Please download the dataset from Google Drive.
https://drive.google.com/drive/folders/1rGoP60VfYkIsTrP5jX7x7XPdkjq8bSiX?usp=sharing



## 2. Run the experiment
To run material extraction, please execute the following command:
```bash
python ffd.py --input_path ./data/mp2v_em_sameIndustry.csv
```
Please note that users can change the input file name to test different HiDy knowledge incorporation.

