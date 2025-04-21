# TPCap


## Training
Train TPCap on the COCO training dataset, using:
```
bash ./scripts/train_tpcap.sh
```

## Evaluation
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
This repo is built on EVCap, we thank the authors for their great effort.