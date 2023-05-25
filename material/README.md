
# Material Knowledge Extraction

In this part, we extract the $Meso$ hierarchy product's upstream/downstream knowledge from Baidu Encyclopedia Corpus. The extraction schema is (product, downstream, material). For example, (A4 paper, downstream, Wood).
## 0. The directory structure of the code
```shell
.
├── paddlenlp         # Please download and install the paddleNLP project and move the paddlenlp folder to this location
├── data              # dataset
├── material_extraction.py       # Do extraction
└── README.md
```

## 1. Download and install the PaddleNLP project
Please follow the PaddleNLP Repository to download the project and install it.
https://github.com/PaddlePaddle/PaddleNLP/



## 2. Do Materials Extraction
To perform material extraction, please execute the following command:
```bash
python material_extraction.py \
    --input_path ./data/20221214.csv \
    --output_path ./data/material_extraction_outputs.csv
```
Please note that the input file should be named as the crawling time for appending the timestamp to append timestamp for the dynamic knowledge.



## For more information
Please refer to this project: https://github.com/PaddlePaddle/PaddleNLP/