
<div align="center">
<h1>HiDy Dataset</h1>


[[Website]](http://143.89.126.57:8003/demo.html)
[[Arxiv Paper]]()
[[Docs]]()
[[Open Database]]()
[[Team]](http://143.89.126.57:8003/fintech.html)


<!-- [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/)](https://pypi.org/project/MineDojo/) -->
[![Status](https://img.shields.io/badge/status-active-success.svg)](https://github.com/K-Quant/HiDy)
[![GitHub Issues](https://img.shields.io/github/issues/K-Quant/HiDy.svg)](https://github.com/K-Quant/HiDy/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/K-Quant/HiDy.svg)](https://github.com/K-Quant/HiDy/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/K-Quant/HiDy.svg)](https://github.com/K-Quant/HiDy/stargazers)
[![GitHub license](https://img.shields.io/github/license/K-Quant/HiDy.svg)](https://github.com/K-Quant/HiDy/blob/main/LICENSE)
[![HitCount](https://views.whatilearened.today/views/github/K-Quant/HiDy.svg)](https://github.com/K-Quant/HiDy)


<p align="center"> A Large-scale Hierarchical Dynamic Financial Knowledge Base</p>

<img src="https://camo.githubusercontent.com/82291b0fe831bfc6781e07fc5090cbd0a8b912bb8b8d4fec0696c881834f81ac/68747470733a2f2f70726f626f742e6d656469612f394575424971676170492e676966" width="800"  height="3">
</div><br>

<img src="images/HiDy_Hierarchy.png" width="1000px">

 HiDy is a hierarchical, dynamic, robust, diverse, and large-scale financial benchmark KB that aims to provide various valuable financial knowledge as critical benchmarking data for fair model testing in different financial tasks. Specifically, HiDy currently contains 34 relation types, more than 493,000 relations, 17 entity types, and more than 51,000 entities. The scale of HiDy is steadily growing due to its continuous updates. To make HiDy easily accessible and retrieved, HiDy is organized in a well-formed financial hierarchy with four branches, *Macro*, *Meso*, *Micro*, and *Others*.
 
With HiDy, users can apply more in-depth, professional, logical, and interpreted knowledge to many common financial tasks, such as stock movement prediction (SMP), financial fraud detection (FFD), supply chain management (SCM), loan default risk prediction (LDRP) and financial event prediction (FEP). 




🎉 **NEWS**: 
- We have open-sourced the `Automatic Knowledge Extraction Package`.
- We have published the 1.0 version of the hierachical dynamic financial knowlegde base `HiDy` in [Zenedo]().




# Contents

- [Installation](#Installation)
- [Getting Started](#Getting-Started)
- [Applications](#Applications)
- [Our Paper](#Check-Out-Our-Paper)
- [License](#License)




# Installation (TODO)

HiDy requires Python ≥ 3.7. We have tested on Ubuntu 20.04 and Mac OS X. 

Download HiDy:

```bash
git clone git@github.com:K-Quant/HiDy.git
cd HiDy
pip install -r requirements.txt
```

To install opennre:

```bash
git clone https://github.com/thunlp/OpenNRE.git --depth 1
```

And then add the follow function to class SentenceRE(nn.Module)
```
    def pred_label(self, eval_loader):
        pred_result = []
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[0]
                args = data[1:]        
                logits = self.parallel_model(*args)
                score, pred = logits.max(-1) # (B)
                # Save result
                for i in range(pred.size(0)):
                    pred_result.append(pred[i].item())
        return pred_result
```
Then modify the original warm up condition in SentenceRE.__init__ to:
```angular2html
        if warmup_step > 0 and train_path != None:
            from transformers import get_linear_schedule_with_warmup
            training_steps = self.train_loader.dataset.__len__() // batch_size * self.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=training_steps)
        else:
            self.scheduler = None
```
Finally, install opennre
```angular2html
pip install -r requirements.txt
python setup.py develop
```
To install keras_contrib:
```
pip install git+https://www.github.com/keras-team/keras-contrib.git
```
**TODO: 预训练模型下载和路径**

To extract the relations from financial news and do the relation fusion, you can run the examples in `update_company_relations`
