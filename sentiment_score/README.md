# Sentiment Score Extraction
## 0. The directory structure of the code

```shell
.
├── OpenNRE         # Please download and install the OpenNRE project and move the folder to this location
└── HiDy           # This is our project
    ├── sentiment_score   # This is the sentiment_score extraction folder
        ├── data         # Dataset
        ├── sentiment_score.py         # To extract sentiment score from reports
        ├── demo.py         # A demo
        └── README.md
    ├── ...
```


## 1. Download and install the PaddleNLP project
Please follow the PaddleNLP Repository to download the project and install it.
https://github.com/PaddlePaddle/PaddleNLP/

## 2.Run the demo
```angular2html
python demo.py \
    --original_report_path ./data/report.json \
    --extracted_data_path ./data/data_extraction.json
```