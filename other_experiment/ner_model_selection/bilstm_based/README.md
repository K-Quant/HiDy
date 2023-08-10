# Named-Entity Recognition(NER) model selection - BiLSTM + CRF, BiLSTM, HMM

In this part, We conducted three experiments to compare the performance of three models on our dataset.


## 0. The directory structure of the code
```shell
.
├── ckpts                   # Saving checkpoints here
├── dataset                 # Data
├── models                  # Model
├── data.py                 # Creating word_lists and tag_lists for NER
├── evaluate.py             # Evaluation
├── evaluating.py           # Evaluation
├── utils.py                # Other common functions
├── experiment.py           # Main
└── README.md
```

## 1. Do BiLSTM + CRF experiments
To perform BiLSTM + CRF method on our data with training, please execute the following command:
```shell
python experiment.py  \
    --BILSTMCRF True \
    --IsLoadBILSTMCRF False 
```

If you do not wish to retrain the model and instead want to directly load a pre-existing model for testing, please execute the following command:
```shell
python experiment.py  \
    --BILSTMCRF True \
    --IsLoadBILSTMCRF True \
    --save_BILSTM_CRF_path ./ckpts/bilstm_crf.pkl
```


## 2. Do BiLSTM experiments
Similarly, to perform BiLSTM method on our data with training, please execute the following command:
```shell
python experiment.py  \
    --BILSTM True \
    --IsLoadBILSTM False 
```

If you do not wish to retrain the model and instead want to directly load a pre-existing model for testing, please execute the following command:
```shell
python experiment.py  \
    --BILSTMCRF True \
    --IsLoadBILSTMCRF True \
    --save_BILSTM_path ./ckpts/bilstm.pkl
```

## 3. Do HMM experiments
Similarly, to perform HMM method on our data with training, please execute the following command:
```shell
python experiment.py  \
    --HMM True \
    --IsLoadHMM False 
```

If you do not wish to retrain the model and instead want to directly load a pre-existing model for testing, please execute the following command:
```shell
python experiment.py  \
    --HMM True \
    --IsLoadHMM True \
    --save_HMM_path ./ckpts/hmm.pkl
```


- ``HMM``: Whether to do the HMM experiment
- ``BILSTM``: Whether to do the BILSTM experiment
- ``BILSTMCRF``: Whether to do the BILSTMCRF experiment
- ``IsLoadHMM``: Whether to load a saved model without the need for retraining
- ``IsLoadBILSTM``: Whether to load a saved model without the need for retraining
- ``IsLoadBILSTMCRF``: Whether to load a saved model without the need for retraining
- ``save_HMM_path``: Path of the saved model.
- ``save_BILSTM_path``: Path of the saved model.
- ``save_BILSTM_CRF_path``: Path of the saved model.