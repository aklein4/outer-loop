
## Data

### Offline RL
- [ ] Format and compile datasets
- [ ] Deduplicate from test data
- [ ] Filter by token length
- [ ] Cluster and shuffle 

### Supervised
- [ ] Find datasets


## Models
- [ ] Llama 3 RoPE frequencies
- [ ] Standard RL Model
    - [ ] Tanh-constrained V-head
- [ ] OLoop RL model
    - [ ] Frozen teacher
    - [ ] SVD Init
    - [ ] Tanh-constrained V-head
- [ ] OLoop Supervised Model
    - [ ] SVD Init


## Training
- [ ] Supervised trainer
- [ ] Offline RL OLoop Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings
    - [ ] Higher lr for new parameters
- [ ] Standard Offline RL Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings
 

## Evaluation

### Benchmarks
- [ ] ManyICLBench (supervised learning)
- [ ] google/rfm-rm-as-user-dataset (personalized RL)
    - [ ] Cluster by similarity
    - [ ] Random RM combination
- [ ] JinaLeejnl/AlignX (personalized RL)
    - [ ] Cluster by personality

### Baselines
- [ ] Strong standard LoRA model 
