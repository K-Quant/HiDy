# Industrial strategy and company_produce_product knowledge extraction

In this part, we extract 1. the $Macro$ hierarchy industrial strategy knowledge from the National Development and Reform Commission Web, and 2. the $Meso$ hierarchy company_produce_product knowledge from the annual report. All annotated data has been provided. The extraction schema is defined as (document_name, mention, industry, date) and (company, produce, product, year). Note that the timestamp of the industrial strategy is obtained through crawling, and we provide it as data. The timestamp of an annual report is the published year of the annual report.

## 0. The directory structure of the code
```shell
.
├── models                  # models
├── data_utils              # Dataloader and evaluation
├── data                    # dataset
├── output                  # the extracted knowledge will be stored here
├── NER.py                  # Named entity extraction
├── requirements.txt        
└── README.md
```

## 1. Install keras-contrib
```angular2html
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

## 2. Do knowledge extraction - industrial strategy 
To perform document_name_mention_industry extraction, please execute the following command:
```shell
python NER.py \
    --input_path ./data/StrategyData/ \
    --entities 'industry' \
    --output_path ./output/macro_s_KB.txt \
    --epochs 1 \
    --whether_train_evaluation False
```

## 3. Do knowledge extraction - annual report
To perform company_produce_product extraction, please execute the following command:
```shell
python NER.py \
    --input_path ./data/AnnualReportData/ \
    --entities 'product' \
    --output_path ./output/meso_ar_KB.txt \
    --epochs 1 \
    --whether_train_evaluation False
```

- ``input_path``: Location of the dataset.
- ``entities``: Entity name.
- ``output_path``: The extracted knowledge will be stored here.
- ``epochs``: Number of epochs.
- ``whether_train_evaluation``: Whether to evaluate the model during the training process.
