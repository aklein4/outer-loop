
# What every handler must do

## Load the dataset
- Load and combine the selected subsets and splits
- inject subset/split identity into each example

## Process each example
- convert into message format
- a handler may emit one output or parallel lists of conversations, latents, and keep flags
- store the latent
    - count number of the kept examples per latent
- determine whether to keep or discard

## Group into horizons
- group by latent
- split into groups of size <=H
    - for groups H//2 <= N < H, repeat to get H
    - discard N < H//2

## Push handled data to a subset
