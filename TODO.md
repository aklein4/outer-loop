
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
- [x] Standard RL Model
    - [x] Tanh-constrained V-head
- [x] OLoop RL model
    - [x] Frozen teacher
    - [x] SVD Init
    - [x] Tanh-constrained V-head
- [x] OLoop Supervised Model
    - [x] SVD Init


## Training
- [ ] Supervised standard trainer
    - [ ] Freeze embeddings
    - [ ] Higher lr for new parameters
- [x] Supervised OLoop trainer
    - [ ] Do not overlap chunks? (Currently overlapping)
    - [x] Freeze embeddings
    - [x] Higher lr for new parameters
- [ ] Offline RL OLoop Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings/teacher
    - [ ] Higher lr for new parameters
- [ ] Standard offline RL trainer
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


## Utilities
- [x] Automatic weight sharding based on shape
