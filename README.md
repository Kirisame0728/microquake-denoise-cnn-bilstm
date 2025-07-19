# CNN-BiLSTM neural network for microquake denoising

## 1.  Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) and requirements
- Download  repository
```bash
git clone https://github.com/Kirisame0728/microquake-denoise-cnn-bilstm.git
cd microquake-denoise-cnn-bilstm
```
- Install to default environment
```bash
conda env update -f=environment.yml -n base
```
- Install the virtual environment
```bash
conda env create -f environment.yml
conda activate dl_env
```
## 2. Pre-trained model
Located in directory: pre_trained/pretrained_denoising_model.pth
