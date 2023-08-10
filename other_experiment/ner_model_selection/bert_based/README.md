# Named-Entity Recognition(NER) model selection - BERT+CRF, BERT+Softmax

In this part, We conducted two experiments to compare the performance of BERT-based named entity recognition models on our dataset.


## 0. The directory structure of the code
```shell
.
├── callback                # Functions related to the training process and Optimizer 
├── datasets                # Data
├── losses                  # Loss
├── metrics                 # Metric
├── models                  # Model
├── outputs                 # Saving models here
├── processors              # Organizing training sequences
├── tools                   # Other common functions
├── experiment.py        
└── README.md
```

## 1. Do BERT+CRF experiments
To perform BERT+CRF method on our data with training, please execute the following command:
```shell
python experiment.py  \
    --model 'bertcrf' \
    --data_dir ./datasets/ \
    --output_dir ./outputs/ \
    --do_train True \
    --num_train_epochs 30 \
    --logging_steps 900 \
    --save_steps 900 
```

If you do not wish to retrain the model and instead want to directly load a pre-existing model for testing, please execute the following command:
```shell
python experiment.py  \
    --model 'bertcrf' \
    --data_dir ./datasets/ \
    --output_dir ./outputs/ \
    --do_train False \
    --load_model_path ./outputs/bertcrf/checkpoint-1800 \
```


## 2. Do BERT+Softmax experiments
Similarly, to perform BERT+Softmax method on our data with training, please execute the following command:
```shell
python experiment.py  \
    --model 'bertsoftmax' \
    --data_dir ./datasets/ \
    --output_dir ./outputs/ \
    --do_train True \
    --num_train_epochs 30 \
    --logging_steps 900 \
    --save_steps 900 
```

If you do not wish to retrain the model and instead want to directly load a pre-existing model for testing, please execute the following command:
```shell
python experiment.py  \
    --model 'bertsoftmax' \
    --data_dir ./datasets/ \
    --output_dir ./outputs/ \
    --do_train False \
    --load_model_path ./outputs/bertsoftmax/checkpoint-1800 \
```


- ``model``: Model name, you can input 'bertsoftmax' or 'bertcrf'
- ``data_dir``: Path of our dataset.
- ``output_dir``: The trained models will be saved here.
- ``do_train``: Whether to do training. If do_train == False, please provide load_model_path.
- ``load_model_path``: Path of the saved model.
- ``num_train_epochs``: Number of epochs.
- ``logging_steps``: Log every X updates steps.
- ``save_steps``: Save checkpoint every X updates steps.


## 3. Acknowledgments
Thanks to https://github.com/lonePatient/BERT-NER-Pytorch/tree/master
