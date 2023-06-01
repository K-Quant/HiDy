# Tax Policy knowledge extraction

In this part, we extract the $Macro$ hierarchy tax policy knowledge from the Ministry of Finance of the PRC website. All annotated data has been provided, you can use this project to extract events from data. You can also obtain the following knowledge directly through annotated data: (tax_policy_name, support_industry, industry, date), (tax_policy_name, tax_cut, policy, date), (tax_policy_name, tax_cut_subject, subject, date), and (tax_policy_name, validity_period, period, date).

## 0. The directory structure of the code
```shell
.
├── data              # dataset
├── checkpoint        # Please download our pre-trained model and place the "checkpoint" folder here
├── evaluate.py       # test the performance
├── utils.py          # data processing
└── README.md
```

## 1. Download and install the PaddleNLP project
Please follow the PaddleNLP Repository to download the project and install it.
https://github.com/PaddlePaddle/PaddleNLP/

## 2. Download the pre-trained model
https://drive.google.com/file/d/1aUnIBieblhjl6uvZ-wEjayV7Z-kkzgpK/view?usp=sharing

## 3. Evaluation
To perform model evaluation, please execute the following command:

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best/checkpoint-100 \
    --test_path ./data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512
```


Description of the evaluation method: 

We employ a single-stage evaluation approach, where tasks such as relation extraction and event extraction, which require sequential prediction, are evaluated separately for each stage of prediction. The validation/testing set by default utilizes all labels at the same level to construct all negative examples. 

You can enable the debug mode to evaluate each positive class separately. Please note that this mode is intended solely for model debugging purposes.

```shell
python evaluate.py \
    --model_path ./checkpoint/model_best/checkpoint-100 \
    --test_path ./data/dev.txt \
    --debug
```


- ``model_path``: The directory path of the model to be evaluated, which should include the model weight file "model_state.pdparams" and the configuration file "config.json". 
- ``test_path``: The file path of the test dataset for evaluation. 
- ``batch_size``: The batch size for processing. Please adjust according to the machine's capacity. The default value is 16. 
- ``max_seq_len``: The maximum length for text segmentation. If the input exceeds this length, it will be automatically segmented. The default value is 512. 
- ``debug``: Whether to enable the debug mode to evaluate each positive class separately. This mode is only intended for model debugging purposes and is disabled by default.

## For more information
Please refer to this project: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie
