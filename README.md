
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

HiDy requires Python ≥ 3.7. We have tested on Ubuntu 20.04 and Mac OS X. **Please follow [this guide](https://docs.minedojo.org/sections/getting_started/install.html#prerequisites)** to install the prerequisites first, such as JDK 8 for running Minecraft backend. We highly recommend creating a new [Conda virtual env](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) to isolate dependencies. Alternatively, we have provided a [pre-built Docker image](https://docs.minedojo.org/sections/getting_started/install.html#docker-image) for easier installation.

Installing HiDy:

```bash
pip install hidy
```

To install the cutting edge version from the main branch of this repo, run:

```bash
git clone https://github.com/DAISYzxy/HiDy && cd HiDy
pip install -e .
```


You can run the script below to verify the installation. It takes a while to compile the Java code for the first time. After that you should see a Minecraft window pop up, with the same gaming interface that human players receive. You should see the message `[INFO] Installation Success` if everything goes well.

```bash
python hidy/scripts/validate_install.py
```
