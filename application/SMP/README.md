# SMP Task Experiment

In this part, we implement the RSR model with HiDy to evaluate the effectiveness of HiDy’s knowledge. Codes are modified from https://github.com/Wentao-Xu/HIST. Please refer to original repo for Qlib data download.

## 0. The directory structure of the code
```shell
.
├── data                # Please download and place dataset here 
├── output              # output
├── dataloader.py       # Dataloader
├── nrsr_learn.py       # NRSR model and experiments
├── rsr_model.py        # RSR model
├── utils.py            # metric and loss
└── README.md
```

## 1. Download and place the data
Please download the dataset from Google Drive.
https://drive.google.com/drive/folders/1B-QJ2Idv4QHrCn0v7WTEm0KOxY-9UP0g?usp=sharing

## 2. Reproduce experiments
The experiment results are stored in output folder, csi300_NRSR_run.log file and info.json. To run the experiments, please execute the following command. 

For industry relation:

```shell
python nrsr_learn.py \
    --stock2stock_matrix ./data/csi300_multi_stock2stock.npy \
    --outdir ./output/NRSR_is
```

For DuEE-Fin + industry relation:

```shell
python nrsr_learn.py \
    --stock2stock_matrix ./data/csi300_multi_dueefin_is.npy \
    --outdir ./output/NRSR_duee
```

For ShanghaiTech + industry relation:

```shell
python nrsr_learn.py \
    --stock2stock_matrix ./data/csi300_multi_sht_is.npy \
    --outdir ./output/NRSR_sht
```

For HiDy relation

```shell
python nrsr_learn.py \
    --stock2stock_matrix ./data/csi300_multi_stock2stock_all.npy \
    --outdir ./output/NRSR_hidy
```


