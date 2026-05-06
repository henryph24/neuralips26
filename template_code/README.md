# Unified Transferability Metrics for Time Series Foundation Models [NeurIPS 2025]

## Model Zoo

We use models from four model families as our model zoo.

| Model Family    | Code                                                         | Pretrained Weight                                            |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MOMENT          | [moment-timeseries-foundation-model/moment: MOMENT: A Family of Open Time-series Foundation Models, ICML'24](https://github.com/moment-timeseries-foundation-model/moment) | [AutonLab (Auton Lab)](https://huggingface.co/AutonLab)      |
| Timer、Timer-XL | [thuml/Large-Time-Series-Model: Official code, datasets and checkpoints for "Timer: Generative Pre-trained Transformers Are Large Time Series Models" (ICML 2024) and subsequent works](https://github.com/thuml/Large-Time-Series-Model) | [thuml/timer-base-84m · Hugging Face](https://huggingface.co/thuml/timer-base-84m) |
| UniTS           | [mims-harvard/UniTS: A unified multi-task time series model.](https://github.com/mims-harvard/UniTS/tree/main) | [Release Pretrained weights · mims-harvard/UniTS](https://github.com/mims-harvard/UniTS/releases/tag/ckpt) |

## Datasets

We follow the experimental settings of TimesNet, and the required datasets can be found in [thuml/Time-Series-Library: A Library for Advanced Deep Time Series Models for General Time Series Analysis.](https://github.com/thuml/Time-Series-Library).

## Usage

You can input features into the TEMPLATE to estimate the transferability score.

```
from utils.metrics import TEMPLATE

dl_score,pl_score,ta_score = TEMPLATE(feature, first_feature,trend_feature)
```

The final transferability score is obtained by normalizing the three scores and then summing them up.

## Citation

If you find this work helpful for your research, please kindly cite the following paper:

```
@inproceedings{zhangTEMPLATE,
  title={Unified Transferability Metrics for Time Series Foundation Models},
  author={Zhang,Weiyang and Chen,Xinyang and Li, Xiucheng and Chen, Kehai and Guan, Weili and Nie, Liqiang},
  booktitle={Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

