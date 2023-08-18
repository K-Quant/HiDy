# Ablation Study for the Financial Fraud Detection(FFD) Task

In this part, we utilize the output of metapath2vec(mp2vec) as node features for each company and further employ three graph neural networks: Graph Convolutional Network(GCN), Graph Attention Network(GAT), and Heterogeneous Graph Attention Network(HAN), for prediction, providing additional benchmark results for researchers who are interested.

The ground truth we can obtain is the company violation records from CSMAR which provide a binary label IsViolated indicating whether fraud occurred. We directly used IsViolated as our labels. In addition, we have uploaded the raw data for your reference: https://docs.google.com/spreadsheets/d/14l7pNbfwQh8WmaWwm0SvEEPeObOxAL5v/edit?usp=sharing&ouid=112799952654350672254&rtpof=true&sd=true.


## 0. The directory structure of the code
```shell
.
├── features            # The output of the metapath2vec(mp2vec), which is used as node feature.
├── GCN_exp                 		
│   ├── config.py               # Args
│   ├── data.py                 # Dataloader
│   ├── layer.py                # Layer
│   ├── sparse_matrices.pkl     # Data
│   ├── model.py                # GCN Model
│   ├── train.py                # Main code
│   ├── utils.py                # Others
├── GAT_exp                 		
│   ├── ckpts              		# Checkpoints
│   ├── layers.py          		# Layer
│   ├── models.py          		# GAT model
│   ├── sparse_matrices.pkl     # Data
│   ├── train.py                # Main code
│   ├── utils.py                # Dataloader, metric, etc
├── HAN_exp                 		
│   ├── sparse_matrices.pkl     # Data
│   ├── model.py                # HAN Model
│   ├── main.py                 # Main code
│   ├── utils.py                # Dataloader, args, etc      
└── README.md
```

## 1.GCN experiments
To perform GCN on our data, please execute the following command under the 'GCN_exp' folder:
```shell
python train.py  \
    --seed 1 \
    --Branch_lst ['IndustryChain','SectorIndustry','Ownership','Partnership']
```


## 2.GAT experiments
To perform GAT on our data, please execute the following command under the 'GAT_exp' folder:
```shell
python train.py  \
    --seed 1 \
    --Branch_lst ['IndustryChain','SectorIndustry','Ownership','Partnership']
```

## 3.HAN experiments
To perform HAN on our data, please execute the following command under the 'HAN_exp' folder:
```shell
python main.py  \
    --seed 1 \
    --Branch_lst ['IndustryChain','SectorIndustry','Ownership','Partnership']
```

- ``seed``: Random seed for repeating experiments.
- ``Branch_lst``: The knowledge you would like to test. Feel free to choose any configuration from this list for testing purposes:
all_test = [
    None,
    ['IndustryChain'], ['SectorIndustry'], ['Ownership'], ['Partnership'],
    ['IndustryChain','SectorIndustry'], ['IndustryChain','Ownership'], ['IndustryChain','Partnership'],
    ['Ownership','SectorIndustry'], ['Partnership','SectorIndustry'], ['Ownership','Partnership'],
    ['Ownership','SectorIndustry','IndustryChain'], ['Partnership','SectorIndustry','IndustryChain'],
    ['Partnership','SectorIndustry','Ownership'], ['Partnership','IndustryChain','Ownership'],
    ['Ownership','IndustryChain','Partnership','SectorIndustry']
]

Note that you can also manually set the other parameters, such as epochs, learning rate (lr), weight decay, hidden layers, and more. For further details, please refer to the code.



## 4. Results
we repeated the experiment ten times by varying the random seed, and subsequently reported the calculated mean and standard deviation of accuracy. The seeds employed for this purpose were 1, 5, 10, 15, 20, 25, 30, 35, 40, and 45. Notably, we utilized the default parameter values as specified in the code.

| Model                | GCN | GAT | HAN |
|----------------------|-----------------|----------------|----------------|
| CSMAR                | 0.6892(0.026)          | 0.6703(0.003)          |0.6783(0.032)|
| CSMAR + IndustryChain| 0.6975(0.027)          | 0.6794(0.010)          |0.7125(0.032)|
| CSMAR + SectorIndustry| 0.6925(0.022)          | 0.6953(0.030)          |0.7031(0.033)|
| CSMAR + Ownership    | 0.6928(0.020)          | 0.6917(0.009)          |0.7061(0.033)|
| CSMAR + Partnership  | 0.6911(0.020)        | 0.6758(0.008)         |0.7244(0.019)|
| CSMAR + IndustryChain + SectorIndustry| 0.7053(0.016)| 0.7042(0.011) |0.7233(0.019)|
| CSMAR + IndustryChain + Ownership    | 0.6925(0.017)| 0.6822(0.008)  |0.7358(0.036)|
| CSMAR + IndustryChain + Partnership  | 0.6942(0.021)| 0.6881(0.014)        |0.7111(0.036)|
| CSMAR + SectorIndustry + Ownership  | 0.6983(0.018)| 0.6958(0.007)  |0.6964(0.022)|
| CSMAR + SectorIndustry  + Partnership|0.6969(0.027)| 0.6875(0.013)    |0.7114(0.022)|
| CSMAR + Ownership + Partnership| 0.6883(0.023)|0.6775(0.007)  |0.7069(0.035)|
| CSMAR + IndustryChain + SectorIndustry + Ownership| 0.7139(0.017)| 0.7069(0.010)   |0.7181(0.028)|
| CSMAR + IndustryChain + SectorIndustry + Partnership|0.7125(0.014)| 0.7119(0.008)     |0.7158(0.028)|
| CSMAR + SectorIndustry + Ownership + Partnership| 0.7025(0.017)| 0.7214(0.015)    |0.7158(0.027)|
| CSMAR + IndustryChain + Ownership + Partnership|0.6947(0.014)| 0.6797(0.008)   |0.7219(0.027)|
| CSMAR + IndustryChain + SectorIndustry + Ownership + Partnership| 0.7056(0.018)|0.7228(0.010)     |0.7314(0.015)|

## 4. Acknowledgments
We use the Pytorch version of the model from the authors. 

Thanks to https://github.com/dragen1860/GCN-PyTorch

Thanks to https://github.com/Diego999/pyGAT

Thanks to https://github.com/dmlc/dgl/tree/master/examples/pytorch/han

