# CenterNet: A PyTorch Re-implementation

This repository contains an unofficial PyTorch re-implementation of the paper: **"CenterNet: Keypoint Triplets for Object Detection"**.

**Link to the original paper:** [https://arxiv.org/abs/2110.06922](https://arxiv.org/abs/1904.08189)

### 3. Configuration

All settings for training are managed in the `configs/red.yaml` file. Before running, make sure to update it with the correct paths and desired hyperparameters, ex - `batch_size`, `learning_rate`, etc. Adjust these as needed for your setup.

### 4. Training

To start training the model, run the `train.py` script:

```bash
python3 train.py
```

**Note on GPUs:** To specify the number of GPUs for training, please modify the relevant parameter directly within the `train.py` script before running it.

### 5. Visualization

During training, the model's performance on the validation set will be visualized automatically.

