# CNN-BiLSTM neural network for microquake denoising

This repository implements the CNN-BiLSTM denoising workflow described in the paper
`Microseismic Data Denoising Using CNN-BiLSTM Model`.

The training pipeline now includes the main paper features that were missing from the
original repository:

- Configurable 1D CNN depth with the paper default of 5 convolutional layers
- Bayesian hyperparameter optimization with a Gaussian-process surrogate and Expected Improvement
- AMP mixed-precision training for lower GPU memory use and faster training on CUDA
- Train/validation/test splitting with the paper default of `70% / 20% / 10%`
- Early stopping, best-checkpoint saving, TensorBoard logging, and test-set evaluation
- Backward-compatible checkpoint loading for new checkpoints, raw `state_dict` files, and older serialized models

## 1. Environment

Clone the repository and create the conda environment:

```bash
git clone https://github.com/Kirisame0728/microquake-denoise-cnn-bilstm.git
cd microquake-denoise-cnn-bilstm
conda env create -f environment.yml
conda activate dl_env
```

## 2. Train the model

Run standard training from the repository root:

```bash
python -m CNN_BiLSTM.train
```

Useful options:

```bash
python -m CNN_BiLSTM.train \
  --num_epochs 30 \
  --batch_size 32 \
  --hidden_size 128 \
  --num_cnn_layers 5 \
  --dropout_rate 0.2 \
  --learning_rate 0.001 \
  --amp
```

Training outputs are saved under `CNN_BiLSTM/logs/<run_name>/`:

- `config.json`
- `history.json`
- `summary.json`
- `best_checkpoint.pth`
- `last_checkpoint.pth`

## 3. Run Bayesian optimization

The paper highlights Bayesian optimization for hyperparameter selection. You can run it with:

```bash
python -m CNN_BiLSTM.bayes_opt --trials 4 --random-starts 2
```

This searches over:

- Number of CNN layers
- BiLSTM hidden size
- Dropout rate
- Learning rate

Search artifacts are written to `CNN_BiLSTM/logs/bayes_search/<run_name>/`.

## 4. Predict / denoise new signals

Use either a new training checkpoint or the provided pretrained file:

```bash
python -m CNN_BiLSTM.predict \
  --model-path pre_trained/pretrained_denoising_model.pth \
  --csv-list data/test_list.csv \
  --data-dir data/test
```

Prediction outputs are saved under `CNN_BiLSTM/results/<run_name>/`.

## 5. Pretrained model

The repository includes a pretrained model at:

```text
pre_trained/pretrained_denoising_model.pth
```

The prediction script can load:

- New checkpoints produced by `CNN_BiLSTM.train`
- Raw `state_dict` files
- Older serialized full-model `.pth` files
