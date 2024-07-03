# VLind-Bench
Paper link: https://arxiv.org/abs/2406.08702
### Usage
1. Download dataset in https://huggingface.co/datasets/klee972/VLind-Bench
2. Directory structure should be as follows.
~~~
├── data
│   ├── data.json
│   ├── counterfactual
│   ├── factual
└── evel
    ├── ctx_cfq
    ├── gpt4o_eval.py
    ├── instructblip_eval.py
    ├── score_pipeline.py
    └── score.sh
~~~
4. Run `gpt4o_eval.py` or `instructblip_eval.py` to generate model predictions.
5. Run `score.sh` to evaluate pipeline scores and accuracies.
