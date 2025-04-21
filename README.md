# TPCap
## Setup
---
1.Install the required packages using conda with the provided environment.yaml file.
2.Preparing the dataset.
3.Download [evaluation.zip](https://github.com/FeiElysia/ViECap/releases/download/checkpoints/evaluation.zip) and [annotations.zip](https://github.com/FeiElysia/ViECap/releases/download/checkpoints/annotations.zip) files, and unzip them to __evaluation__ and __data__ (you should create them first).
```
mkdir ./evaluation
unzip evaluation.zip -d ./evaluation
mkdir ./data
unzip evaluation.zip -d ./data
```
4.Download LLaVA pretrained projector [Vicuna-7B-v1.3](https://huggingface.co/liuhaotian/llava-pretrain-vicuna-7b-v1.3)

## Training
Train TPCap on the COCO training dataset, using:
```
bash ./scripts/train_tpcap.sh
```

## Evaluation
---
Evaluate the trained TPCap on the COCO test set, NoCaps validation set, Flickr30k test set, and WHOOPS, using the following script:
```
bash ./scripts/eval_tpcap_coco.sh coco 0
bash ./scripts/eval_tpcap_flickr30k.sh flickr30k 0
bash ./scripts/eval_tpcap_nocaps.sh nocaps 0
bash ./scripts/eval_tpcap_whoops.sh whoops 0
```

## Citation
---
```
@article{zhang2025tpcap,
  title={Tpcap: Unlocking zero-shot image captioning with trigger-augmented and multi-modal purification modules},
  author={Zhang, Ruoyu and Wang, Lulu and He, Yi and Pan, Tongling and Yu, Zhengtao and Li, Yingna},
  journal={arXiv preprint arXiv:2502.11024},
  year={2025}
}
```

## Acknowledgements
---
This repo is built on [EVCap](https://github.com/Jiaxuan-Li/EVCap), and we thank the authors for their great effort.
The evaluation and data files are from [ViECap](https://github.com/FeiElysia/ViECap), and we thank the authors for their work.
