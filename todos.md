## V7 Model
- [x] Push the current code to V6
- [x] Introduct Concept of Example
- [x] Change augmentation to be based on the example, as opposed to the task
- [x] Synthetic data needs to have uniform embeddings (This is important. Do this for SYNTH/REARC + Tama)
- [x] Create tokeniser.py and add interpreter tokenizer
- [x] Create new dataset.py to handle new ProgramDataset
- [x] Add Tama dataset
- [x] Fix encoder attention (Seems to throw no nans anymore magically)
- [x] Encoder-Decoder Attention Should be different
- [x] Add loop_heads
- [x] Specify various configs
- [x] Rename to REPL
- [x] Create the multi-level loss module
- [x] Investigate why there are nans in inp_kv_cache (new issue for loop encoder kv cache) (This is expected)
- [x] Rope Should be different for LoopEncoder, rotation should take bigger jumpd?
    - Added a new Rope2D to handle grids
- [x] Optimise Rope2D, add decorators, deal with device, etc. (register buffer should move to correct device as it is added to state_dict)
- [x] Fix the array toknization to allow Rope2D
- [x] Remove Causal_Out from Model_Input
- [x] Change the collate_fn to include position indices
- [x] Make padding as the zeroth token always
- [x] enc_dec_mask is incorrect. It should include all the input tokens (including non-grid tokens)
- [x] Change model forward to take predicted output (which is shifted by one inside the model)
- [x] Seprate out program embedding from invariant transformation
- [x] Add type embeddings to the tokens
- [x] setting attn_mask to None has no effect on StateAggregator. (This is expected in the kv_cached forwardx model because due to caching, the past tokens
don't access future tokens making it causal by default)
- [x] Implement return kv_cache in the forward of the model
- [x] Add multi-level loss to the model
- [x] Non-grid tokens should have no rope2d applied. Currently it applies (0, 0) which is incorrect!! (There is restoring of original embedding. Tested OK!)
- [x] implement forwardx in the model
- [x] Make loop encoder incremental

- [x] Test Incremental Decoding
- [x] Port greedy search to the new model
- [x] Make model scriptable
- [x] Verify Rope2D is working
- [x] Test and add multilevel loss with/to repl model

- [x] Add training only methods to the model
- [x] Add copy program embedding from source to target method
- [x] Migrate Hparams to the new model
- [x] Update ArcTrainer
- [x] Add loop level token/sample accuracy
- [x] Add difficulty based metric
- [x] Consider replacing MSE to KL Divergence?? (No need. Remove this metric tracking altogether)
- [x] Add the abiltiy to specify eval batch size
- [x] L2 regularisation despite setting the norm to 1.0
- [x] Update Solver
- [x] Add synthetic tasks
- [x] Remove any samples with > max_seq_len input or output
- [x] Let the training begin
- [x] Add fork support and resume the previous training (with export CUDA_LAUNCH_BLOCKING=1)

- [x] Test training on a single batch?
- [x] Change state_agg positional embedding to use Rope Instead. (Not needed. I have changed to use GRU which is much faster)
- [x] Measure the impact of wandb logging. Remove excessive logging. Adds 8-10 ms. I am going to keep it
- [x] Add min max option num examples option to the data
- [x] Will it be faster to apply log head in one go? (Yes!)
- [x] Make data loading astonishingly fast 
- [x] Cache data for training
- [x] Refresh data every epoch. This should substantially reduce the dataset size. Added option to refresh data all the time
- [x] Make learning rate decay forever (There seems to be a trend where the progress on the accuracy becomes linear as the learning rate flattens out)
- [x] Test torch scripted model for training. The speed is not different.
- [x] Why is there no REARC in the training models? (It was not included in train_collection :facepalm:)
- [x] Try resuming training with new data
- [x] Something wrong with either tokenAcc or SampleAcc or both as I noticed SampleAcc would go to 100 while TokenAcc still remained 28%. Need to investigate.
        - There was a bug in tokenacc computation.
- [x] Try training only the ARC_TRAIN dataset (without any auxiliary tasks). Use high regularisation. It trains but doesn't get to high enough accuracy


- [x] Add the ARCSynthTasks dataset to the training mix
- [ ] Detangle the datasets. It seems combining ARC with REARC is hurting performance on ARC. Add prefix to ARC datasets. Also add a plug for arc synth tasks
- [ ] Make program vocab size additive




- [ ] Try learning a single task on a really small network (with lots of augmentations)

## New Training
- [ ] Overfitting so use higher regularisation
- [ ] Number of params in the model (vs Computation Equivalent Params)

- [ ] Do a single batch test and then see if see if permuting the input changes the output. All valid inputs should affect the output, but not the invalid tokens
- [ ] Use ARC verifiers to generate novel valid programs

- [ ] Port beam search to the new model

- [ ] Ability to specify start step?
- [x] (Top-K progressive loss + exponential weighted?) Implemented MultiLevelLoss
- [x] Add the augmentation scale to different datasets (Archived in favour of num of examples per task)

import ipdb; ipdb.set_trace()

# V2 TODOS
- [x] Move current code to a separate branch
- [x] Generate Synthetic ARC Dataset (Identity, Transform, CP)
- [x] Create task difficulty Metric
- [x] Change the training to include another parameter to specify levels and training data
- [x] Move to Swiglu Activation
- [x] Move to RMS Norm
- [x] Move to RoPe embeddings
- [x] Add autocast + no_grad to rotary embeddings
- [x] Don't share the weights between the input embedding and output embedding?
- [x] Move to transformer++ architecture
- [x] Simplify recurrence to include FFN also
- [x] Change the model to the new architecture
- [x] Removed all the bias from the network
- [x] Changed to default initialization
- [x] Fix the network initialization
- [x] Remove stale code
- [x] Adjust the checkpoint/model load code to the new architecture
- [x] Make the solver.py run
- [x] Remove tensorboard from the code
- [x] Might need to fix the from_dict (hparams)
- [x] Make run id as the hash of the hparams (archived as it makes it difficult because one can't recreate a deleted run)
- [x] Change the grouping order of the metrics in wandb
- [x] Add WandB resume functionality
- [x] Difference run for the dev mode
- [x] Add epoch level to tensorboard
- [x] Add metadata to the batch so analysis can be done during training
- [x] Add various Histograms
- [x] Analyse weird rank histogram distribution. (Most likely due to small sample size)
- [x] Break down accuracy by levels (Added granular ranks)
- [x] Visualise program embeddings
- [x] Add model gradients
- [x] Add Scaling synthetic data?
- [x] Track best accuracy and checkpoint (and delete rest to save space)
- [x] Add best accuracy summary to the wandb
- [x] Add drop out to prevent overfitting
- [x] Read grokking papers
- [x] Add Grokfast to the model
- [x] Add lindecay lr schedule
- [x] Is there a reason to have min_lr instead of letting cosine annealing to go to 0?. Yes, otherwise it rises up again
- [x] Add warm up to constant LR schedule to all
- [x] Log param norms (with reduced frequency)
- [x] Implement decreasing linear LR
- [x] Shorten the DIVA dataset
- [x] Log accuracy by dataset and level
- [x] Log accuracy by Rank/Level
- [x] Try only on the ARC dataset to see the effect of auxiliary tasks
- [x] Convert into encoder-decoder architecture
- [x] Change the program code for synthetic data to be task independent (decided not to because of scale up)
- [x] [LOOPED TRANSFORMERS ARE BETTER AT LEARNINGLEARNING ALGORITHMS](https://arxiv.org/pdf/2311.12424)
- [x] [End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking](https://arxiv.org/abs/2202.05826)
- [x] Implement Input injection
- [x] Implement truncated looped transformer backprop
- [x] design loop curriculum
- [x] Implement progressive loss
- [x] Log convergence of output wrt to loop
- [x] Implement fork functionality allowing to fork a training (resume + override + new run)
- [x] Forking does not change the LR initial learning rate, this needs to be fixed!
    - https://discuss.pytorch.org/t/optimizer-lr-changing-after-loading-state-dict/86285

- [x] Forking does not change the weight decay. Need to fix that!
- [x] Prediction is dependent on input size. Need to fix that. Provide a variable mask. Modify input batching.
- [x] Also randomize the input order for each epoch. Introduce noise in length
- [x] Check that output doesn't change by modifying the input size.
- [x] Fix the downstream metrics, etc in ArcTrainer.py calculations to account for above dataset collate_fn changes
- [x] Allow for bigger batch size by periodically clearing the cache
- [x] Remove IDENT_CID tranformation in favour of `original` version
- [x] Check if the results of the new model match for greedy evaluation
- [x] Add Beam Search (inefficient implementation)
- [x] Test KV cache implementation
- [x] Implement Gredy search with and without KV cache and verify the results match
- [x] Fix the beam search implementation
- [x] Test the beam search implementation speed on GPU
- [x] Implement parallel model executor
- [x] Implement LazyAdamW + L1 regularisation (inside the LazyAdamW) (Should improve memory + speed (check this) (And use https://github.com/davda54/lazy-adam/blob/master/lazy_adam.py) (check if fused can be used!))
- [x] Set sparse=True for embedding and migrate to LazwAdamW
- [x] Add l1_coeff param to the solver
- [x] Migrate solver to interpreter2
- [x] Implement embedding norm setting to 1.0
- [x] Log embedding sparsity to wandb

- [x] Add idenity block expansion
- [x] Add train only identity block facility in solver train new
- [ ] Add train only identity block facility in solver train fork (This doesn't make sense. Allow specifying starting step offset)
- [x] There is potentially a problem with loading model from a checkpoint in train new. try A2D5M128B2H8L8_w7o_pin with train new 
    - May be it is fine. It could be down to model sensitive to batch size early on in the training (remember there is noise now)

## V6 Analysis
- [x] Create a task learner/solver
- [x] Analyse trained model L2 norms (nothing to gained here)
- [x] Visualise embeddings of the trained model. There does not seem to be any clear semantic similarity between the embeddings
- [ ] Check the difference between learned embedding between train and test (is there a significant difference?)
- [ ] Check if the learned model does well on Tama?
- [ ] Check solved model embeddings on ARC_EVAL
- [ ] Visualise attention?
- [ ] Fine-tune ARC_EVAL on the trained model?




- [ ] Verify that it learns for ARC_TRAIN
- [ ] Test performance of the model ARC_EVAL



- [ ] Write analysis code topk vs beam width vs accuracy vs iters
- [x] Make all of the interpreter torch scripting compatible
- [ ] [Block expansion paper](https://arxiv.org/abs/2401.02415)

- [ ] Add weighting param to each Task Loader based on their importance. 
    - Check how Tamas dataset performs when not trained on it
- [ ] Change batch size for eval?
- [ ] Change in data breaks forked training. Need to fix that

## Future Work
- [ ] More data https://github.com/mxgmn/WaveFunctionCollapse
- [ ] Can the current problem be formulated as a RL problem? The inverse loss being the reward
- [x] Implement faster kv-cached version of the model
- [ ] Visualise attention to make sure that program embedding is referenced across the loops
- [ ] https://github.com/neoneye/arc-dataset-tama/tree/main
- [ ] Use Simon's https://github.com/neoneye/simon-arc-lab/tree/main/simon_arc_dataset_run  to generate more data
- [ ] Investigate REARC dataset and generation process (possibly use embeddings to selectively generate)
- [ ] Write Evaluation + Visualisation Script
- [ ] Symbolic generator https://github.com/richemslie/McARGA
- [ ] Make curriculum resume easier?
- [ ] Start using Gumbel Softmax (Make it switchable hyperparameter)
- [ ] Start using MetaLearning
- [ ] Implement MoE (Mixture of Experts) (dependent on the program encoder)
- [ ] Read about Latformer https://infoscience.epfl.ch/entities/publication/89c8f500-c685-42b9-83cf-d55666c6afdc
- [ ] Should switch to graph NN? 
- [ ] Can we measure grokking in the hidden phase
- [ ] Once it works, may be introduce neural stack (transducers)

## Evaluation

### ARC_TRAIN
- [x] Fine tune program embedding on ARC_TRAIN
    - when mlr == 0.0, set requires_grad = False for the rest of the model

- [ ] Measure the impact of beam search on the accuracy
    - [ ] Given eval example, embedding and loops, generate possible outputs
    - [ ] Save visualisations of the outputs

- [ ] Measure the impact of loops on the accuracy. Could it be that higher levels just need more loops??
    - [ ] plot against the levels to ensure changing loops doesn't change the accuracy of the lower levels

## ARC_EVAL
- [ ] Ability to Specify dataset in solver.py
- [ ] Repeat for ARC_EVAL



## Future Reads
- [ ] [Looped Transformers as Programmable Computers](https://arxiv.org/pdf/2301.13196)]
- [ ] [Trace Dataset](https://arc-visualizations.github.io/index.html)
- [ ] [Larc DSL to assist training](https://arxiv.org/pdf/2106.07824)
# Grokking Ideas
- Start out with out augmentation 
    - Helps with overfitting

- [ ] [The AdEMAMix Optimizer](https://arxiv.org/abs/2409.03137)
- [ ] Improving Transformer Models by Reordering their Sublayers
- [ ] https://arxiv.org/pdf/2409.04777

# V2 Later  
- [ ] The plan is to see if there is natural clustering of the program embeddings.
If they do, then it makes sense to build a V3 with incremental collapse of nearby embedding
to build a program library. At that time, it could make sense to have program embeddings that are different for each layer. As always, previously solved programs will have identity added in next levels.

The big idea is that for simple programs (like identity, rotate, etc) we know the embedding clusters. For complex program, we can build new clusters to build core program library.
But this is all done in the future. For now, we will just use the unique embedding for all programs, and analyse the clusters


## Difficulty Metric
- Size
- Size Scale
- Histogram Diff
- Color Variablity
- Compression Ratio
- Entropy
- Number of Objects
- Number of Colors




# V1 TODOS
- [x] Wrap with torch.autocast even during evaluation
- [x] Load model from another model
- [x] Add BatchSize metric (in Tokens)
- [x] Weird formatting Tokens(/s)
- [x] Add Trainer.state_dict() and Trainer.load_state_dict()
- [x] Make trainer take trainer config and nothing else
- [x] Add CLI
- [x] If resume, do extra checks and use special copy!
- [x] Add --checkpint option to train new
- [x] Add gitcommmit to hparams for the purposes of reproduction
- [x] Add script to sync runs from remote to local
- [x] Figure out how to copy the checkpoint from remote to local
- [x] Test training on the GPU
- [x] Make evaluation run right before training starts (always?)
- [x] Print checkpoint path when loading model from a checkpoint
- [x] Create simplied experiment running with automatic run name
- [x] Allow for easy lr_finding (without needed to delete the run)
- [x] Allow for easy test run (for GPU)
- [x] Set up logger across the board
- [x] Allow for easy compute bump
- [x] Allow easy lr change
- [x] Consider alternative learning rates per epoch? (Training mode)
- [x] For follow ups, take run path instead to use autocomplete
- [x] Add BS size, and seq_len to the metrics
- [x] Make the naming of follow modelsize experiment consistent with lr followup
- [x] Test out the changes on the Lightning GPU
- [x] Run 128 dim training on H100!!
- [ ] Add identity program and augment data further
- [x] Make sync script to sync continuously
- [ ] Analyse trained model on the dataset
- [ ] AvgSeqLen for the correct samples (to see whether that is changing even if the sample accuracy goes down) (May be this can be done as part of analysis as opposed to metric during training?)
- [ ] Run Cuda Empty Cache After each train and eval epoch


# Experimentation Strategy
- Find_L
- After training stagnates
    - Find a new LR on the loaded model
    - Increase n_mixers (until the loss stops getting better)
    - Increase n_layers (until the loss stops getting better)

## Start with dim 64, heads 8, layers 3, mixers 3, blocks 3


## Move to dim 128, heads 16, layers 3, mixers 3, blocks 3
- Load the model from the previous checkpoint everytime (FirstExperiment in lightning_runs)
- Start with dim 128, heads 16, layers 3, mixers 3, blocks 3
- Find_LR
- After training stagnates
    - Find a new LR on the loaded model
    - Increase n_mixers (until the loss stops getting better)
    - Increase n_layers (until the loss stops getting better)


# GPU Stats

## A10G (125 BFLOAT16, 24GB RAM)
## Model Set up 
N_LAYERS = 3
N_MIXERS = 3
N_BLOCKS = 3
N_HEADS = 16
N_DIM = 128


### Fixed BS, Fixed SEQ_LEN

#### Config 1
BS = 256
SEQ_LEN = 1024
DYNAMMIC_BATCHING = False
PIN_MEMORY = False
USE_COMPILE = False

Using fused AdamW: True
step     0 | loss: 2.228072 | dt: 2101.40ms | tok/sec: 124747.25 | BS: 256 | SL:  1024 | TOKENS: 262144
step     1 | loss: 1.859667 | dt: 1484.73ms | tok/sec: 176560.02 | BS: 256 | SL:  1024 | TOKENS: 262144
step     2 | loss: 0.985434 | dt: 1484.70ms | tok/sec: 176563.48 | BS: 256 | SL:  1024 | TOKENS: 262144
step     3 | loss: 10.429130 | dt: 1484.90ms | tok/sec: 176539.47 | BS: 256 | SL:  1024 | TOKENS: 262144
step     4 | loss: 1.804607 | dt: 1484.43ms | tok/sec: 176595.95 | BS: 256 | SL:  1024 | TOKENS: 262144
step     5 | loss: 0.801844 | dt: 1484.73ms | tok/sec: 176560.44 | BS: 256 | SL:  1024 | TOKENS: 262144
step     6 | loss: 0.924060 | dt: 1484.89ms | tok/sec: 176541.37 | BS: 256 | SL:  1024 | TOKENS: 262144
step     7 | loss: 1.029959 | dt: 1484.92ms | tok/sec: 176537.26 | BS: 256 | SL:  1024 | TOKENS: 262144
step     8 | loss: 1.226858 | dt: 1484.68ms | tok/sec: 176565.52 | BS: 256 | SL:  1024 | TOKENS: 262144
step     9 | loss: 1.323041 | dt: 1485.44ms | tok/sec: 176476.25 | BS: 256 | SL:  1024 | TOKENS: 262144
step    10 | loss: 1.234716 | dt: 1484.82ms | tok/sec: 176549.73 | BS: 256 | SL:  1024 | TOKENS: 262144


#### Config 2

BS = 128. # Out of memory for BS 256
SEQ_LEN = 1024
DYNAMMIC_BATCHING = False
PIN_MEMORY = True
USE_COMPILE = True

Using fused AdamW: True
step     0 | loss: 2.018222 | dt: 39537.33ms | tok/sec: 3315.15 | BS: 128 | SL:  1024 | TOKENS: 131072
step     1 | loss: 1.600374 | dt: 696.27ms | tok/sec: 188247.91 | BS: 128 | SL:  1024 | TOKENS: 131072
step     2 | loss: 10.744127 | dt: 696.13ms | tok/sec: 188286.72 | BS: 128 | SL:  1024 | TOKENS: 131072
step     3 | loss: 1.911825 | dt: 696.17ms | tok/sec: 188276.34 | BS: 128 | SL:  1024 | TOKENS: 131072
step     4 | loss: 0.791202 | dt: 696.25ms | tok/sec: 188253.58 | BS: 128 | SL:  1024 | TOKENS: 131072
step     5 | loss: 1.109717 | dt: 696.26ms | tok/sec: 188252.81 | BS: 128 | SL:  1024 | TOKENS: 131072
step     6 | loss: 0.983122 | dt: 696.47ms | tok/sec: 188193.59 | BS: 128 | SL:  1024 | TOKENS: 131072
step     7 | loss: 1.259393 | dt: 696.03ms | tok/sec: 188312.97 | BS: 128 | SL:  1024 | TOKENS: 131072
step     8 | loss: 0.826588 | dt: 696.24ms | tok/sec: 188257.13 | BS: 128 | SL:  1024 | TOKENS: 131072
step     9 | loss: 1.086343 | dt: 696.34ms | tok/sec: 188230.96 | BS: 128 | SL:  1024 | TOKENS: 131072
step    10 | loss: 0.953750 | dt: 696.30ms | tok/sec: 188240.76 | BS: 128 | SL:  1024 | TOKENS: 131072
step    11 | loss: 0.782584 | dt: 696.48ms | tok/sec: 188193.20 | BS: 128 | SL:  1024 | TOKENS: 131072
step    12 | loss: 0.823419 | dt: 696.34ms | tok/sec: 188228.90 | BS: 128 | SL:  1024 | TOKENS: 131072
step    13 | loss: 1.022881 | dt: 696.04ms | tok/sec: 188311.04 | BS: 128 | SL:  1024 | TOKENS: 131072
step    14 | loss: 0.793522 | dt: 696.17ms | tok/sec: 188275.57 | BS: 128 | SL:  1024 | TOKENS: 131072
step    15 | loss: 0.834600 | dt: 696.47ms | tok/sec: 188195.39 | BS: 128 | SL:  1024 | TOKENS: 131072
step    16 | loss: 0.863483 | dt: 695.92ms | tok/sec: 188344.52 | BS: 128 | SL:  1024 | TOKENS: 131072
step    17 | loss: 0.816272 | dt: 696.44ms | tok/sec: 188202.74 | BS: 128 | SL:  1024 | TOKENS: 131072
step    18 | loss: 0.834148 | dt: 696.70ms | tok/sec: 188132.02 | BS: 128 | SL:  1024 | TOKENS: 131072
step    19 | loss: 0.844696 | dt: 695.91ms | tok/sec: 188346.33 | BS: 128 | SL:  1024 | TOKENS: 131072
step    20 | loss: 0.941007 | dt: 696.43ms | tok/sec: 188206.54 | BS: 128 | SL:  1024 | TOKENS: 131072


# Config 3
Config Above + max-autotune

SingleProcess AUTOTUNE takes 3.8676 seconds
step     0 | loss: 2.502519 | dt: 128844.13ms | tok/sec: 1017.29 | BS: 128 | SL:  1024 | TOKENS: 131072
step     1 | loss: 1.244585 | dt: 2613.97ms | tok/sec: 50142.92 | BS: 128 | SL:  1024 | TOKENS: 131072
step     2 | loss: 10.890123 | dt: 684.79ms | tok/sec: 191405.63 | BS: 128 | SL:  1024 | TOKENS: 131072
step     3 | loss: 1.356171 | dt: 684.90ms | tok/sec: 191373.51 | BS: 128 | SL:  1024 | TOKENS: 131072
step     4 | loss: 1.150334 | dt: 685.16ms | tok/sec: 191301.66 | BS: 128 | SL:  1024 | TOKENS: 131072
step     5 | loss: 1.498256 | dt: 684.60ms | tok/sec: 191458.62 | BS: 128 | SL:  1024 | TOKENS: 131072
step     6 | loss: 1.458305 | dt: 685.07ms | tok/sec: 191327.36 | BS: 128 | SL:  1024 | TOKENS: 131072
step     7 | loss: 1.361034 | dt: 684.74ms | tok/sec: 191419.89 | BS: 128 | SL:  1024 | TOKENS: 131072
step     8 | loss: 1.280503 | dt: 684.99ms | tok/sec: 191348.13 | BS: 128 | SL:  1024 | TOKENS: 131072
step     9 | loss: 0.759941 | dt: 685.31ms | tok/sec: 191260.33 | BS: 128 | SL:  1024 | TOKENS: 131072
step    10 | loss: 0.857740 | dt: 684.76ms | tok/sec: 191413.82 | BS: 128 | SL:  1024 | TOKENS: 131072
step    11 | loss: 0.868535 | dt: 684.98ms | tok/sec: 191352.73 | BS: 128 | SL:  1024 | TOKENS: 131072
step    12 | loss: 0.831517 | dt: 684.93ms | tok/sec: 191366.52 | BS: 128 | SL:  1024 | TOKENS: 131072
step    13 | loss: 0.945244 | dt: 684.97ms | tok/sec: 191355.59 | BS: 128 | SL:  1024 | TOKENS: 131072
step    14 | loss: 0.877321 | dt: 684.94ms | tok/sec: 191362.65 | BS: 128 | SL:  1024 | TOKENS: 131072
step    15 | loss: 0.836476 | dt: 684.87ms | tok/sec: 191381.24 | BS: 128 | SL:  1024 | TOKENS: 131072
step    16 | loss: 0.649172 | dt: 685.13ms | tok/sec: 191310.31 | BS: 128 | SL:  1024 | TOKENS: 131072
step    17 | loss: 0.873301 | dt: 685.16ms | tok/sec: 191300.79 | BS: 128 | SL:  1024 | TOKENS: 131072

## Config 4
BS = 186
SEQ_LEN = 1024
DYNAMMIC_BATCHING = False
PIN_MEMORY = False
USE_COMPILE = True

Using fused AdamW: True
step     0 | loss: 2.157411 | dt: 39092.31ms | tok/sec: 4872.16 | BS: 186 | SL:  1024 | TOKENS: 190464
step     1 | loss: 1.796013 | dt: 1003.93ms | tok/sec: 189718.34 | BS: 186 | SL:  1024 | TOKENS: 190464
step     2 | loss: 1.345305 | dt: 1003.38ms | tok/sec: 189823.02 | BS: 186 | SL:  1024 | TOKENS: 190464
step     3 | loss: 1.267480 | dt: 1003.59ms | tok/sec: 189782.89 | BS: 186 | SL:  1024 | TOKENS: 190464
step     4 | loss: 1.603131 | dt: 1004.22ms | tok/sec: 189664.34 | BS: 186 | SL:  1024 | TOKENS: 190464
step     5 | loss: 1.330172 | dt: 1003.95ms | tok/sec: 189714.78 | BS: 186 | SL:  1024 | TOKENS: 190464
step     6 | loss: 0.951018 | dt: 1003.76ms | tok/sec: 189750.29 | BS: 186 | SL:  1024 | TOKENS: 190464
step     7 | loss: 1.091412 | dt: 1003.63ms | tok/sec: 189775.63 | BS: 186 | SL:  1024 | TOKENS: 190464
step     8 | loss: 0.773648 | dt: 1003.73ms | tok/sec: 189755.66 | BS: 186 | SL:  1024 | TOKENS: 190464
step     9 | loss: 0.862764 | dt: 1003.74ms | tok/sec: 189754.17 | BS: 186 | SL:  1024 | TOKENS: 190464
step    10 | loss: 0.932378 | dt: 1003.78ms | tok/sec: 189747.50 | BS: 186 | SL:  1024 | TOKENS: 190464


# Config 5
BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = False
PIN_MEMORY = True
USE_COMPILE = True

SingleProcess AUTOTUNE takes 3.8679 seconds
step     0 | loss: 1.987013 | dt: 106809.80ms | tok/sec: 1227.15 | BS: 128 | SL:  1024 | TOKENS: 131072
step     1 | loss: 1.610485 | dt: 2659.35ms | tok/sec: 49287.23 | BS: 128 | SL:  1024 | TOKENS: 131072
step     2 | loss: 1.999692 | dt: 685.28ms | tok/sec: 191266.91 | BS: 128 | SL:  1024 | TOKENS: 131072
step     3 | loss: 1.907133 | dt: 685.21ms | tok/sec: 191285.95 | BS: 128 | SL:  1024 | TOKENS: 131072
step     4 | loss: 2.249970 | dt: 684.91ms | tok/sec: 191370.31 | BS: 128 | SL:  1024 | TOKENS: 131072
step     5 | loss: 2.027641 | dt: 685.15ms | tok/sec: 191303.19 | BS: 128 | SL:  1024 | TOKENS: 131072
step     6 | loss: 1.590154 | dt: 684.92ms | tok/sec: 191368.78 | BS: 128 | SL:  1024 | TOKENS: 131072
step     7 | loss: 1.834768 | dt: 684.89ms | tok/sec: 191376.24 | BS: 128 | SL:  1024 | TOKENS: 131072
step     8 | loss: 1.491631 | dt: 685.05ms | tok/sec: 191331.82 | BS: 128 | SL:  1024 | TOKENS: 131072
step     9 | loss: 1.337023 | dt: 684.96ms | tok/sec: 191358.12 | BS: 128 | SL:  1024 | TOKENS: 131072
step    10 | loss: 1.168574 | dt: 685.11ms | tok/sec: 191316.64 | BS: 128 | SL:  1024 | TOKENS: 131072
step    11 | loss: 0.880955 | dt: 685.10ms | tok/sec: 191316.84 | BS: 128 | SL:  1024 | TOKENS: 131072
step    12 | loss: 1.194465 | dt: 684.93ms | tok/sec: 191365.12 | BS: 128 | SL:  1024 | TOKENS: 131072


# Config 6 (Move data to GPU)
BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = False
PIN_MEMORY = False
USE_COMPILE = True
DATA_DEVICE = device

Using fused AdamW: True
step     0 | loss: 2.636904 | dt: 37833.91ms | tok/sec: 3464.41 | BS: 128 | SL:  1024 | TOKENS: 131072
step     1 | loss: 1.094191 | dt: 2585.73ms | tok/sec: 50690.55 | BS: 128 | SL:  1024 | TOKENS: 131072
step     2 | loss: 8.306509 | dt: 685.36ms | tok/sec: 191246.42 | BS: 128 | SL:  1024 | TOKENS: 131072
step     3 | loss: 1.226206 | dt: 684.90ms | tok/sec: 191373.44 | BS: 128 | SL:  1024 | TOKENS: 131072
step     4 | loss: 0.953235 | dt: 684.70ms | tok/sec: 191431.09 | BS: 128 | SL:  1024 | TOKENS: 131072
step     5 | loss: 1.205658 | dt: 684.48ms | tok/sec: 191490.10 | BS: 128 | SL:  1024 | TOKENS: 131072
step     6 | loss: 0.965219 | dt: 684.73ms | tok/sec: 191421.49 | BS: 128 | SL:  1024 | TOKENS: 131072
step     7 | loss: 1.039164 | dt: 684.70ms | tok/sec: 191430.69 | BS: 128 | SL:  1024 | TOKENS: 131072
step     8 | loss: 0.894166 | dt: 684.58ms | tok/sec: 191463.15 | BS: 128 | SL:  1024 | TOKENS: 131072
step     9 | loss: 0.880680 | dt: 684.48ms | tok/sec: 191490.36 | BS: 128 | SL:  1024 | TOKENS: 131072
step    10 | loss: 0.743279 | dt: 684.87ms | tok/sec: 191383.57 | BS: 128 | SL:  1024 | TOKENS: 131072
step    11 | loss: 0.957619 | dt: 684.66ms | tok/sec: 191441.29 | BS: 128 | SL:  1024 | TOKENS: 131072

### Dynamic Batching

## Config 1
BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = True
PIN_MEMORY = False
USE_COMPILE = False
DATA_DEVICE = torch.device('cpu')

Using fused AdamW: True
step     0 | loss: 2.569271 | dt: 1047.58ms | tok/sec: 125126.20 | BS: 1160 | SL:   113 | TOKENS: 131080
step     1 | loss: 4.171596 | dt: 507.36ms | tok/sec: 258285.16 | BS: 361 | SL:   363 | TOKENS: 131043
step     2 | loss: 4.501309 | dt: 657.59ms | tok/sec: 199274.36 | BS: 180 | SL:   728 | TOKENS: 131040
step     3 | loss: 3.080075 | dt: 612.15ms | tok/sec: 214329.25 | BS: 222 | SL:   591 | TOKENS: 131202
step     4 | loss: 3.052235 | dt: 664.40ms | tok/sec: 197365.80 | BS: 186 | SL:   705 | TOKENS: 131130
step     5 | loss: 2.732193 | dt: 563.96ms | tok/sec: 232515.28 | BS: 282 | SL:   465 | TOKENS: 131130
step     6 | loss: 3.034506 | dt: 688.96ms | tok/sec: 190540.46 | BS: 198 | SL:   663 | TOKENS: 131274
step     7 | loss: 3.273795 | dt: 604.54ms | tok/sec: 216759.98 | BS: 210 | SL:   624 | TOKENS: 131040
step     8 | loss: 2.687103 | dt: 745.43ms | tok/sec: 176089.48 | BS: 167 | SL:   786 | TOKENS: 131262
step     9 | loss: 2.853419 | dt: 473.25ms | tok/sec: 276809.49 | BS: 524 | SL:   250 | TOKENS: 131000
step    10 | loss: 2.481112 | dt: 699.52ms | tok/sec: 187757.63 | BS: 203 | SL:   647 | TOKENS: 131341
step    11 | loss: 2.580350 | dt: 660.55ms | tok/sec: 198817.18 | BS: 256 | SL:   513 | TOKENS: 131328


## Config 2

BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = True
PIN_MEMORY = False
USE_COMPILE = False
DATA_DEVICE = device


Restarted cloudspace (Python)

Using device: cuda
230
Using fused AdamW: True
step     0 | loss: 2.880877 | dt: 1251.80ms | tok/sec: 104729.44 | BS: 228 | SL:   575 | TOKENS: 131100
step     1 | loss: 5.574933 | dt: 516.58ms | tok/sec: 253965.07 | BS: 387 | SL:   339 | TOKENS: 131193
step     2 | loss: 7.059329 | dt: 727.91ms | tok/sec: 180263.72 | BS: 161 | SL:   815 | TOKENS: 131215
step     3 | loss: 6.560645 | dt: 677.82ms | tok/sec: 193052.62 | BS: 193 | SL:   678 | TOKENS: 130854
step     4 | loss: 5.202521 | dt: 461.53ms | tok/sec: 284010.86 | BS: 1160 | SL:   113 | TOKENS: 131080
step     5 | loss: 4.969984 | dt: 563.19ms | tok/sec: 232656.44 | BS: 283 | SL:   463 | TOKENS: 131029
step     6 | loss: 4.195243 | dt: 546.77ms | tok/sec: 239662.81 | BS: 455 | SL:   288 | TOKENS: 131040
step     7 | loss: 3.790758 | dt: 580.59ms | tok/sec: 225857.94 | BS: 310 | SL:   423 | TOKENS: 131130
step     8 | loss: 3.623176 | dt: 695.10ms | tok/sec: 188825.66 | BS: 201 | SL:   653 | TOKENS: 131253
step     9 | loss: 3.459740 | dt: 546.16ms | tok/sec: 240373.78 | BS: 261 | SL:   503 | TOKENS: 131283
step    10 | loss: 3.290764 | dt: 567.61ms | tok/sec: 230904.04 | BS: 508 | SL:   258 | TOKENS: 131064
step    11 | loss: 3.094271 | dt: 744.08ms | tok/sec: 175736.08 | BS: 167 | SL:   783 | TOKENS: 130761
step    12 | loss: 2.935688 | dt: 633.17ms | tok/sec: 207030.91 | BS: 2913 | SL:    45 | TOKENS: 131085
step    13 | loss: 2.628853 | dt: 454.77ms | tok/sec: 288314.55 | BS: 1066 | SL:   123 | TOKENS: 131118
step    14 | loss: 2.851003 | dt: 633.14ms | tok/sec: 207038.09 | BS: 2913 | SL:    45 | TOKENS: 131085
step    15 | loss: 2.880949 | dt: 597.73ms | tok/sec: 219583.86 | BS: 209 | SL:   628 | TOKENS: 131252
step    16 | loss: 2.937217 | dt: 499.32ms | tok/sec: 262331.13 | BS: 342 | SL:   383 | TOKENS: 130986
step    17 | loss: 2.989309 | dt: 627.67ms | tok/sec: 208677.71 | BS: 236 | SL:   555 | TOKENS: 130980

Average Tokens/Sec 221552.37575654936

## Config 3
BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = True
PIN_MEMORY = False
USE_COMPILE = True
DATA_DEVICE = device

Using fused AdamW: True
step     0 | loss: 2.819807 | dt: 60601.13ms | tok/sec: 2162.34 | BS: 480 | SL:   273 | TOKENS: 131040
step     1 | loss: 5.590381 | dt: 709.99ms | tok/sec: 184813.57 | BS: 161 | SL:   815 | TOKENS: 131215
step     2 | loss: 5.879003 | dt: 549.35ms | tok/sec: 238578.37 | BS: 508 | SL:   258 | TOKENS: 131064
step     3 | loss: 5.092117 | dt: 727.31ms | tok/sec: 180705.57 | BS: 167 | SL:   787 | TOKENS: 131429
step     4 | loss: 9.320673 | dt: 631.53ms | tok/sec: 207478.00 | BS: 179 | SL:   732 | TOKENS: 131028
step     5 | loss: 5.037078 | dt: 709.53ms | tok/sec: 184479.30 | BS: 161 | SL:   813 | TOKENS: 130893
step     6 | loss: 5.527558 | dt: 562.97ms | tok/sec: 232924.52 | BS: 310 | SL:   423 | TOKENS: 131130
step     7 | loss: 4.522044 | dt: 562.18ms | tok/sec: 233251.98 | BS: 310 | SL:   423 | TOKENS: 131130
step     8 | loss: 3.529794 | dt: 614.75ms | tok/sec: 213233.14 | BS: 2913 | SL:    45 | TOKENS: 131085
step     9 | loss: 3.250527 | dt: 543.59ms | tok/sec: 241230.74 | BS: 282 | SL:   465 | TOKENS: 131130
step    10 | loss: 3.089163 | dt: 517.23ms | tok/sec: 253401.59 | BS: 434 | SL:   302 | TOKENS: 131068
step    11 | loss: 3.397029 | dt: 692.01ms | tok/sec: 188819.68 | BS: 155 | SL:   843 | TOKENS: 130665
step    12 | loss: 3.192613 | dt: 576.78ms | tok/sec: 227177.46 | BS: 207 | SL:   633 | TOKENS: 131031
step    13 | loss: 3.237758 | dt: 442.90ms | tok/sec: 295960.09 | BS: 1160 | SL:   113 | TOKENS: 131080
step    14 | loss: 2.701633 | dt: 456.24ms | tok/sec: 287388.49 | BS: 1273 | SL:   103 | TOKENS: 131119
step    15 | loss: 2.608603 | dt: 726.96ms | tok/sec: 180562.25 | BS: 167 | SL:   786 | TOKENS: 131262
step    16 | loss: 2.653764 | dt: 726.01ms | tok/sec: 180108.95 | BS: 167 | SL:   783 | TOKENS: 130761
step    17 | loss: 3.089550 | dt: 1159.70ms | tok/sec: 110704.71 | BS: 8024 | SL:    16 | TOKENS: 128384
step    18 | loss: 2.717544 | dt: 583.13ms | tok/sec: 225042.34 | BS: 212 | SL:   619 | TOKENS: 131228
step    19 | loss: 2.652747 | dt: 691.77ms | tok/sec: 189107.77 | BS: 155 | SL:   844 | TOKENS: 130820
step    20 | loss: 2.752538 | dt: 624.28ms | tok/sec: 210092.83 | BS: 247 | SL:   531 | TOKENS: 131157
step    21 | loss: 2.756042 | dt: 527.77ms | tok/sec: 248748.36 | BS: 261 | SL:   503 | TOKENS: 131283
step    22 | loss: 2.491590 | dt: 467.15ms | tok/sec: 280378.34 | BS: 577 | SL:   227 | TOKENS: 130979
step    23 | loss: 2.482580 | dt: 679.34ms | tok/sec: 192761.53 | BS: 150 | SL:   873 | TOKENS: 130950
step    24 | loss: 2.565932 | dt: 673.94ms | tok/sec: 194083.26 | BS: 200 | SL:   654 | TOKENS: 130800
step    25 | loss: 2.483941 | dt: 514.18ms | tok/sec: 254947.24 | BS: 427 | SL:   307 | TOKENS: 131089
step    26 | loss: 2.479700 | dt: 476.03ms | tok/sec: 275183.14 | BS: 615 | SL:   213 | TOKENS: 130995
step    27 | loss: 2.683478 | dt: 548.33ms | tok/sec: 239006.93 | BS: 506 | SL:   259 | TOKENS: 131054
step    28 | loss: 2.595172 | dt: 650.39ms | tok/sec: 202050.07 | BS: 188 | SL:   699 | TOKENS: 131412
step    29 | loss: 2.559697 | dt: 666.58ms | tok/sec: 197005.34 | BS: 196 | SL:   670 | TOKENS: 131320
step    30 | loss: 2.599407 | dt: 631.64ms | tok/sec: 207157.57 | BS: 179 | SL:   731 | TOKENS: 130849
step    31 | loss: 2.603373 | dt: 631.52ms | tok/sec: 207198.40 | BS: 179 | SL:   731 | TOKENS: 130849
step    32 | loss: 2.444568 | dt: 494.90ms | tok/sec: 264899.84 | BS: 690 | SL:   190 | TOKENS: 131100
step    33 | loss: 2.483058 | dt: 553.26ms | tok/sec: 236933.91 | BS: 971 | SL:   135 | TOKENS: 131085
step    34 | loss: 2.489186 | dt: 528.94ms | tok/sec: 247559.73 | BS: 264 | SL:   496 | TOKENS: 130944
step    35 | loss: 2.520271 | dt: 681.79ms | tok/sec: 191794.96 | BS: 204 | SL:   641 | TOKENS: 130764
step    36 | loss: 2.604440 | dt: 452.38ms | tok/sec: 289734.94 | BS: 514 | SL:   255 | TOKENS: 131070
step    37 | loss: 2.425594 | dt: 534.33ms | tok/sec: 245492.98 | BS: 477 | SL:   275 | TOKENS: 131175
step    38 | loss: 2.550474 | dt: 458.38ms | tok/sec: 285738.92 | BS: 539 | SL:   243 | TOKENS: 130977
step    39 | loss: 2.505481 | dt: 490.80ms | tok/sec: 266867.88 | BS: 370 | SL:   354 | TOKENS: 130980
step    40 | loss: 2.319757 | dt: 502.22ms | tok/sec: 261237.53 | BS: 400 | SL:   328 | TOKENS: 131200


Average Tokens/Sec 220053.52472468527

## Config 4 (with max-autotune)
BS = 128
SEQ_LEN = 1024
DYNAMMIC_BATCHING = True
PIN_MEMORY = False
USE_COMPILE = True
DATA_DEVICE = device


step    21 | loss: 2.700026 | dt: 578.32ms | tok/sec: 226259.79 | BS: 217 | SL:   603 | TOKENS: 130851
step    22 | loss: 2.876824 | dt: 522.01ms | tok/sec: 251204.22 | BS: 470 | SL:   279 | TOKENS: 131130
step    23 | loss: 2.935159 | dt: 651.88ms | tok/sec: 201028.29 | BS: 193 | SL:   679 | TOKENS: 131047
step    24 | loss: 2.610232 | dt: 638.12ms | tok/sec: 205494.05 | BS: 186 | SL:   705 | TOKENS: 131130
step    25 | loss: 2.759628 | dt: 668.48ms | tok/sec: 196344.90 | BS: 201 | SL:   653 | TOKENS: 131253
step    26 | loss: 2.383958 | dt: 612.42ms | tok/sec: 214406.25 | BS: 173 | SL:   759 | TOKENS: 131307
step    27 | loss: 2.529367 | dt: 650.91ms | tok/sec: 201031.03 | BS: 193 | SL:   678 | TOKENS: 130854
step    28 | loss: 2.475479 | dt: 624.52ms | tok/sec: 209825.12 | BS: 180 | SL:   728 | TOKENS: 131040
step    29 | loss: 2.447816 | dt: 683.50ms | tok/sec: 191397.07 | BS: 155 | SL:   844 | TOKENS: 130820
step    30 | loss: 2.619485 | dt: 510.46ms | tok/sec: 256730.79 | BS: 804 | SL:   163 | TOKENS: 131052
step    31 | loss: 2.573774 | dt: 2522.74ms | tok/sec: 51979.23 | BS: 186 | SL:   705 | TOKENS: 131130
step    32 | loss: 2.606685 | dt: 2485.02ms | tok/sec: 52839.40 | BS: 173 | SL:   759 | TOKENS: 131307
step    33 | loss: 2.656848 | dt: 504.41ms | tok/sec: 260022.94 | BS: 767 | SL:   171 | TOKENS: 131157
step    34 | loss: 2.454620 | dt: 553.44ms | tok/sec: 236936.93 | BS: 310 | SL:   423 | TOKENS: 131130
step    35 | loss: 2.547375 | dt: 700.85ms | tok/sec: 186762.51 | BS: 161 | SL:   813 | TOKENS: 130893
step    36 | loss: 2.522463 | dt: 2539.43ms | tok/sec: 51602.13 | BS: 180 | SL:   728 | TOKENS: 131040
step    37 | loss: 2.527239 | dt: 478.87ms | tok/sec: 273716.70 | BS: 662 | SL:   198 | TOKENS: 131076
step    38 | loss: 2.515349 | dt: 658.01ms | tok/sec: 199571.65 | BS: 196 | SL:   670 | TOKENS: 131320
step    39 | loss: 2.429265 | dt: 760.47ms | tok/sec: 171763.30 | BS: 140 | SL:   933 | TOKENS: 130620
step    40 | loss: 2.529031 | dt: 570.75ms | tok/sec: 229596.27 | BS: 209 | SL:   627 | TOKENS: 131043
step    41 | loss: 2.942672 | dt: 707.56ms | tok/sec: 185251.27 | BS: 3972 | SL:    33 | TOKENS: 131076
step    42 | loss: 2.294428 | dt: 517.58ms | tok/sec: 253703.47 | BS: 259 | SL:   507 | TOKENS: 131313
step    43 | loss: 2.454246 | dt: 486.91ms | tok/sec: 269096.77 | BS: 382 | SL:   343 | TOKENS: 131026
step    44 | loss: 2.134202 | dt: 535.68ms | tok/sec: 244792.08 | BS: 282 | SL:   465 | TOKENS: 131130
step    45 | loss: 2.526522 | dt: 2549.47ms | tok/sec: 51391.78 | BS: 174 | SL:   753 | TOKENS: 131022
step    46 | loss: 2.548441 | dt: 448.56ms | tok/sec: 291994.85 | BS: 539 | SL:   243 | TOKENS: 130977
step    47 | loss: 2.162262 | dt: 2477.27ms | tok/sec: 52933.26 | BS: 282 | SL:   465 | TOKENS: 131130
step    48 | loss: 2.354471 | dt: 575.18ms | tok/sec: 228208.76 | BS: 334 | SL:   393 | TOKENS: 131262
step    49 | loss: 2.443728 | dt: 2647.49ms | tok/sec: 49390.61 | BS: 167 | SL:   783 | TOKENS: 130761
step    50 | loss: 2.478902 | dt: 703.33ms | tok/sec: 186340.52 | BS: 162 | SL:   809 | TOKENS: 131058
Average Tokens/Sec 159780.92950690264



## Dynamic SEQ_LEN , and fixed BS

BS = 128
SEQ_LEN = None
DYNAMMIC_BATCHING = False
PIN_MEMORY = True
USE_COMPILE = False
DATA_DEVICE = torch.device('cpu')


step    37 | loss: 0.924907 | dt: 480.27ms | tok/sec: 201755.12 | BS: 128 | SL:   757 | TOKENS: 96896
step    38 | loss: 0.874325 | dt: 725.98ms | tok/sec: 164499.58 | BS: 128 | SL:   933 | TOKENS: 119424
step    39 | loss: 0.948443 | dt: 478.61ms | tok/sec: 201381.85 | BS: 128 | SL:   753 | TOKENS: 96384
step    40 | loss: 1.049533 | dt: 593.03ms | tok/sec: 182170.35 | BS: 128 | SL:   844 | TOKENS: 108032
step    41 | loss: 0.895600 | dt: 593.71ms | tok/sec: 181744.73 | BS: 128 | SL:   843 | TOKENS: 107904
step    42 | loss: 0.822723 | dt: 585.87ms | tok/sec: 178060.62 | BS: 128 | SL:   815 | TOKENS: 104320
step    43 | loss: 0.824941 | dt: 585.93ms | tok/sec: 178042.58 | BS: 128 | SL:   815 | TOKENS: 104320
step    44 | loss: 0.976548 | dt: 602.13ms | tok/sec: 185582.31 | BS: 128 | SL:   873 | TOKENS: 111744
step    45 | loss: 0.826149 | dt: 717.94ms | tok/sec: 160816.66 | BS: 128 | SL:   902 | TOKENS: 115456
step    46 | loss: 0.854831 | dt: 717.90ms | tok/sec: 161003.50 | BS: 128 | SL:   903 | TOKENS: 115584
step    47 | loss: 0.921785 | dt: 717.58ms | tok/sec: 160896.60 | BS: 128 | SL:   902 | TOKENS: 115456
step    48 | loss: 0.951824 | dt: 479.72ms | tok/sec: 201985.85 | BS: 128 | SL:   757 | TOKENS: 96896
step    49 | loss: 0.746024 | dt: 718.07ms | tok/sec: 160965.87 | BS: 128 | SL:   903 | TOKENS: 115584
step    50 | loss: 0.893467 | dt: 726.42ms | tok/sec: 164401.10 | BS: 128 | SL:   933 | TOKENS: 119424
Average Tokens/Sec 176726.92224335115