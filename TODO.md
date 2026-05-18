
## Data

### Offline RL
- [ ] Format and compile datasets
- [ ] Deduplicate from test data
- [ ] Filter by token length
- [ ] Cluster and shuffle 

### Supervised
- [ ] Find datasets


## Models
- [x] Llama 3 RoPE frequencies
- [ ] Standard RL Model
    - [ ] Tanh-constrained V-head
- [x] OLoop RL model
    - [x] Frozen teacher
    - [x] SVD Init
    - [x] Tanh-constrained V-head
- [x] OLoop Supervised Model
    - [x] SVD Init


## Training
- [ ] Supervised trainer
- [ ] Offline RL OLoop Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings/teacher
    - [ ] Higher lr for new parameters
- [ ] Standard Offline RL Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings/teacher
 

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


### Utilities
- [ ] Automatic weight sharding based on shape
