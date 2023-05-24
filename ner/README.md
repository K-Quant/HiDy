# Industrial strategy and company_produce_product knowledge extraction

In this part, we extract 1. the $Macro$ hierarchy industrial strategy knowledge from the National Development and Reform Commission Web, and 2. the $Meso$ hierarchy company_produce_product knowledge from the annual report. All annotated data has been provided. The extraction schema is defined as (document_name, mention, industry, date) and (company, produce, product, year).

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

## 1. Do knowledge extraction - industrial strategy 
To perform document_name_mention_industry extraction, please execute the following command:
```shell
python NER.py \
    --input_path ./data/StrategyData/ \
    --entities 'industry' \
    --output_path ./output/macro_s_KB.txt \
    --epochs 1 \
    --whether_train_evaluation False
```

## 2. Do knowledge extraction - annual report
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
