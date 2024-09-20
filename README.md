# ArcSolver
This is my attempt to solving arc puzzles using Deep Learning.

## Innovation Tracker
- Task Augmentation
- Program Embedding
- Loops
- Progressive Loss
- Decoder only Encoder-Decoder architecture
- Input Injection
- LazyW optimizer
- Embedding L1 Regularization
- Embedding Norm Pinning
- Beam / Greedy Search (cached)
- Scripted model for speed
- Identity Block Expansion

## WIP Innovation 
- Embedding Sparsity (Via forced sparsity)

## Version 2 Training Observations
-  Noticed that higher recurrence leads to better generalisation. Performs better on eval (but not on train)
    - https://wandb.ai/abhilab-hobby/baseline4dimv2/runs/v1.2.2?nw=nwuserabhilab
    - https://wandb.ai/abhilab-hobby/baseline4dimv2/runs/v1.2?nw=nwuserabhilab