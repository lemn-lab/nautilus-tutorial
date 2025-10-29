# nautilus-tutorial
Material for the Nautilus Tutorial

To train a toy GPT-2 from scratch

```python3
python3 gpt2_training.py --shakespeare_files dataset/input.txt
```

To train a toy GPT-2 from scratch with bayesian hyperparameter sweep

```python3
python3 gpt2_training_sweep.py --shakespeare_files dataset/input.txt
```
To train a toy GPT-2 from scratch with bayesian hyperparameter sweep on two GPUs (DDP)

```python3
torchrun --nproc_per_node=2 gpt2_training_sweep_ddp.py --shakespeare_files dataset/input.txt
```