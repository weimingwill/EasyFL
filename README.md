<div align="center">
  <img src="docs/en/_static/image/easyfl-logo.png" width="500"/>
  <h1 align="center">EasyFL: A Low-code Federated Learning Platform</h1>

[![PyPI](https://img.shields.io/pypi/v/easyfl)](https://pypi.org/project/easyfl)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://easyfl.readthedocs.io/en/latest/)
[![license](https://img.shields.io/github/license/easyfl-ai/easyfl.svg)](https://github.com/easyfl-ai/easyfl/blob/master/LICENSE)
[![maintained](https://img.shields.io/badge/Maintained%3F-YES-yellow.svg)](https://github.com/easyfl-ai/easyfl/graphs/commit-activity)
[![Downloads](https://pepy.tech/badge/easyfl)](https://pepy.tech/project/easyfl)

[📘 Documentation](https://easyfl.readthedocs.io/en/latest/) | [🛠️ Installation](https://easyfl.readthedocs.io/en/latest/get_started.html)
</div>

## Introduction

**EasyFL** is an easy-to-use federated learning (FL) platform based on PyTorch. It aims to enable users with various levels of expertise to experiment and prototype FL applications with little/no coding. 

You can use it for:
* FL Research on algorithm and system
* Proof-of-concept (POC) of new FL applications
* Prototype of industrial applications
* Learning FL implementations

We currently focus on horizontal FL, supporting both cross-silo and cross-device FL. You can learn more about federated learning from these [resources](https://github.com/weimingwill/awesome-federated-learning#blogs). 

## Major Features

**Easy to Start**

EasyFL is easy to install and easy to learn. It does not have complex dependency requirements. You can run EasyFL on your personal computer with only three lines of code ([Quick Start](docs/en/quick_run.md)).

**Out-of-the-box Functionalities**

EasyFL provides many out-of-the-box functionalities, including datasets, models, and FL algorithms. With simple configurations, you simulate different FL scenarios using the popular datasets. We support both statistical heterogeneity simulation and system heterogeneity simulation.

**Flexible, Customizable, and Reproducible**

EasyFL is flexible to be customized according to your needs. You can easily migrate existing CV or NLP applications into the federated manner by writing the PyTorch codes that you are most familiar with. 

**Multiple Training Modes**

EasyFL supports **standalone training**, **distributed training**, and **remote training**. By developing the code once, you can easily speed up FL training with distributed training on multiple GPUs. Besides, you can even deploy it to Kubernetes with Docker using remote training.

## Getting Started

You can refer to [Get Started](docs/en/get_started.md) for installation and [Quick Run](docs/en/quick_run.md) for the simplest way of using EasyFL.

For more advanced usage, we provide a list of tutorials on:
* [High-level APIs](docs/en/tutorials/high-level_apis.md)
* [Configurations](docs/en/tutorials/config.md)
* [Datasets](docs/en/tutorials/dataset.md)
* [Models](docs/en/tutorials/model.md)
* [Customize Server and Client](docs/en/tutorials/customize_server_and_client.md)
* [Distributed Training](docs/en/tutorials/distributed_training.md)
* [Remote Training](docs/en/tutorials/remote_training.md)


## Projects & Papers

We have released the source code for the following papers under the `applications` folder:

- FedSSL: [[code]](https://github.com/EasyFL-AI/EasyFL/tree/master/applications/fedssl) for two papers: [Divergence-aware Federated Self-Supervised Learning](https://openreview.net/forum?id=oVE1z8NlNe) (_ICLR'2022_)  and [Collaborative Unsupervised Visual Representation Learning From Decentralized Data](https://openaccess.thecvf.com/content/ICCV2021/html/Zhuang_Collaborative_Unsupervised_Visual_Representation_Learning_From_Decentralized_Data_ICCV_2021_paper.html) (_ICCV'2021_)
- FedReID: [[code]](https://github.com/EasyFL-AI/EasyFL/tree/master/applications/fedreid) for two papers: [Performance Optimization for Federated Person Re-identification via Benchmark Analysis](https://dl.acm.org/doi/10.1145/3394171.3413814) (_ACMMM'2020_) and [Optimizing Performance of Federated Person Re-identification: Benchmarking and Analysis](https://dl.acm.org/doi/10.1145/3531013) (_TOMM_)
- FedUReID: [[code]](https://github.com/EasyFL-AI/EasyFL/tree/master/applications/fedureid) for [Joint Optimization in Edge-Cloud Continuum for Federated Unsupervised Person Re-identification](https://arxiv.org/abs/2108.06493) (_ACMMM'2021_)


:bulb: We will release the source codes of these projects in this repository. Please stay tuned.

We have been doing research on federated learning for several years, the following are our additional publications.

- EasyFL: A Low-code Federated Learning Platform For Dummies, _IEEE Internet-of-Things Journal_. [[paper]](https://arxiv.org/abs/2105.07603)
- Federated Unsupervised Domain Adaptation for Face Recognition, _ICME'22_. [[paper]](https://weiming.me/publication/fedfr/)
- Optimizing Federated Unsupervised Person Re-identification via Camera-aware Clustering, _MMSP'22_. [[paper]](https://ieeexplore.ieee.org/abstract/document/9949249/)

## Join Our Community

Please join our community on Slack: [easyfl.slack.com](https://easyfl.slack.com) 

We will post updated features and answer questions on Slack.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this platform or related projects in your research, please cite this project.

```
@article{zhuang2022easyfl,
  title={Easyfl: A low-code federated learning platform for dummies},
  author={Zhuang, Weiming and Gan, Xin and Wen, Yonggang and Zhang, Shuai},
  journal={IEEE Internet of Things Journal},
  year={2022},
  publisher={IEEE}
}
```

## Main Contributors

Weiming Zhuang [:octocat:](https://github.com/weimingwill) <br/>
Xin Gan [:octocat:](https://github.com/codergan)
