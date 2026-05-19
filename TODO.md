
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

### Long-context
- [x] Supervised long-context standard trainer
    - [x] Freeze embeddings
- [x] Supervised long-context OLoop trainer
    - [ ] Do not overlap chunks? (Currently overlapping)
    - [ ] Fix Llama expecting BOS token at later chunks
    - [x] Freeze embeddings
    - [x] Higher lr for new parameters

### Offline RL
- [ ] Offline RL OLoop Trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings/teacher
    - [ ] Higher lr for new parameters
- [ ] Standard offline RL trainer
    - [ ] Optional KL regularization
    - [ ] Freeze embeddings/teacher

### Supervised
- [ ] Supervised standard trainer
    - [ ] Freeze embeddings
- [ ] Supervised OLoop trainer
    - [ ] Freeze embeddings
    - [ ] Higher lr for new parameters
 

## Evaluation

### Baselines
- [ ] Strong standard LoRA model

### Benchmarks

#### Offline RL
- [ ] google/rfm-rm-as-user-dataset (personalized RL)
    - [ ] Cluster by similarity
    - [ ] Random RM combination
- [ ] JinaLeejnl/AlignX (personalized RL)
    - [ ] Cluster by personality

#### Supervised
- [ ] ManyICLBench (supervised learning)


## Utilities
- [x] Automatic weight sharding based on shape
