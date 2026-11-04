# IMRouter: Imbalance-Aware Temporal Pattern Routing for LLM-Based Time Series Forecasting

This repository provides the implementation of **IMRouter**, an imbalance-aware framework for LLM-based time-series forecasting. IMRouter introduces pattern-adaptive temporal routing and routing-guided semantic alignment to reduce the disproportionate influence of frequent temporal patterns and improve forecasting performance on rare temporal regimes.

The released code supports:

- long-term forecasting;
- short-term forecasting on M4;
- few-shot forecasting on the ETT benchmarks.

## 1. Repository Structure

```text
IMRouter/
├── data_provider/                  # Dataset loading and preprocessing
├── exp/                            # Training and evaluation pipelines
├── models/                         # IMRouter and GPT-2 backbone modules
├── scripts/
│   ├── long_term_forecasting/      # Long-term forecasting scripts
│   ├── short_term_forecasting/     # M4 forecasting scripts
│   └── few_shot_forecasting/       # Few-shot forecasting scripts
├── utils/                          # Metrics, losses, and utility functions
├── datasets/                       # Benchmark datasets
├── gpt2/                           # Local GPT-2 configuration and weights
├── checkpoints/                    # Saved checkpoints
├── results/                        # Prediction results
├── pca.py                          # Principal word-embedding extraction
├── run.py                          # Main entry point
├── requirements.txt
└── README.md
```

For compatibility with the inherited codebase, the current implementation is registered under the model identifier `CALF`. Therefore, the provided scripts use `--model CALF`, although the implemented method is IMRouter.

## 2. Environment Setup

We recommend creating an isolated Python environment:

```bash
conda create -n imrouter python=3.9 -y
conda activate imrouter
```

Install a PyTorch build compatible with the local CUDA environment, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

All commands below should be executed from the repository root.

## 3. Dataset Preparation

The benchmark datasets can be obtained by following the data-preparation instructions and download resources provided in the official **One Fits All** repository:

https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All

After downloading and extracting the datasets, place them under the `datasets/` directory using the following structure:

```text
datasets/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── electricity/
│   └── electricity.csv
├── traffic/
│   └── traffic.csv
├── weather/
│   └── weather.csv
└── m4/
    ├── M4-info.csv
    ├── Yearly-train.csv
    ├── Yearly-test.csv
    ├── Quarterly-train.csv
    ├── Quarterly-test.csv
    ├── Monthly-train.csv
    ├── Monthly-test.csv
    ├── Weekly-train.csv
    ├── Weekly-test.csv
    ├── Daily-train.csv
    ├── Daily-test.csv
    ├── Hourly-train.csv
    ├── Hourly-test.csv
    └── submission-Naive2.csv
```

The dataset paths used by each experiment are specified through `--root_path` and `--data_path` in the corresponding shell scripts.

## 4. GPT-2 Weight Preparation

IMRouter uses the pretrained GPT-2 backbone. Download the GPT-2 configuration and PyTorch weights from the Hugging Face model repository and save them in a local `gpt2/` directory:

https://huggingface.co/openai-community/gpt2

The following command downloads the model and stores it in the required location:

```bash
mkdir -p gpt2

python - <<'PY'
from transformers import GPT2Model

model = GPT2Model.from_pretrained("openai-community/gpt2")
model.save_pretrained("./gpt2", safe_serialization=False)

print("GPT-2 has been saved to ./gpt2")
PY
```

After downloading, the directory should contain at least:

```text
gpt2/
├── config.json
└── pytorch_model.bin
```

The model code loads the backbone through:

```python
GPT2Model.from_pretrained("gpt2")
```

Therefore, experiments should be launched from the repository root so that the local `gpt2/` directory can be resolved correctly.

## 5. Principal Word-Embedding Preparation

The cross-modal alignment module uses principal components extracted from the GPT-2 word-token embedding matrix. After preparing the local GPT-2 weights, run:

```bash
python pca.py
```

This command generates:

```text
wte_pca_500.pt
```

in the repository root. Ensure that this file is available before training.

## 6. Long-Term Forecasting

The long-term forecasting scripts are provided in:

```text
scripts/long_term_forecasting/
```

For example, run IMRouter on ETTh1 with:

```bash
bash scripts/long_term_forecasting/ETTh1.sh
```

Other supported benchmarks can be launched using:

```bash
bash scripts/long_term_forecasting/ETTh2.sh
bash scripts/long_term_forecasting/ETTm1.sh
bash scripts/long_term_forecasting/ETTm2.sh
bash scripts/long_term_forecasting/weather.sh
bash scripts/long_term_forecasting/electricity.sh
bash scripts/long_term_forecasting/traffic.sh
```

The default input length is 96, and the forecasting horizons are configured in the corresponding scripts.

## 7. Few-Shot Forecasting

Few-shot experiments use a restricted proportion of the available training data. The scripts are located in:

```text
scripts/few_shot_forecasting/
```

For example:

```bash
bash scripts/few_shot_forecasting/ETTm1.sh
```

The complete set of available scripts is:

```bash
bash scripts/few_shot_forecasting/ETTh1.sh
bash scripts/few_shot_forecasting/ETTh2.sh
bash scripts/few_shot_forecasting/ETTm1.sh
bash scripts/few_shot_forecasting/ETTm2.sh
```

The training-data percentage and other experimental settings can be modified directly in each shell script.

## 8. Short-Term Forecasting

Short-term forecasting is evaluated on the M4 benchmark. Run:

```bash
bash scripts/short_term_forecasting/m4.sh
```

The script evaluates the Yearly, Quarterly, Monthly, Weekly, Daily, and Hourly subsets.

## 9. Outputs

By default, the experimental pipeline stores outputs in the following locations:

```text
checkpoints/     # Model checkpoints
results/         # Predictions and numerical results
```

Additional task-level summaries may be written to text files in the repository root. The exact output paths are controlled by `run.py` and the experiment classes under `exp/`.

## 10. Reproducibility Notes

- The default random seed is fixed to `2021` in `run.py`.
- GPU selection is controlled through `CUDA_VISIBLE_DEVICES` in each shell script.
- Dataset paths, batch sizes, learning rates, forecasting horizons, expert configurations, and loss coefficients are specified in the experiment scripts.
- The `gpt2/` directory and `wte_pca_500.pt` must be prepared before training.
- All commands should be executed from the repository root to preserve relative-path resolution.

## 11. Acknowledgements

This implementation is developed based on the CALF forecasting framework and incorporates components from established time-series forecasting repositories. We gratefully acknowledge the authors of the following open-source projects:

- [CALF](https://github.com/Hank0626/CALF)
- [One Fits All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Autoformer](https://github.com/thuml/Autoformer)

Users should follow the licenses and terms of use associated with the original codebases, benchmark datasets, and pretrained GPT-2 weights.
