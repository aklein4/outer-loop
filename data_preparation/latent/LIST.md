
## Sequence to Sequence
(Gradient clustering unless otherwise noted)

### Post-Training
 - [x] nvidia/Nemotron-Post-Training-Dataset-v2
 - [x] nvidia/Llama-Nemotron-Post-Training-Dataset
 - [ ] OpenDataArena/OpenDataArena-scored-data # not all good quality?

### Chat
 - [x] HuggingFaceTB/smoltalk
 - [x] HuggingFaceTB/smoltalk2 -> SFT
 - [x] teknium/OpenHermes-2.5
 - [x] microsoft/orca-agentinstruct-1M-v1
 - [ ] IgnoraZ/SynthQuestions -> synthquestions, realquestions # problem with jsonl formatting
 - [x] TIGER-Lab/WebInstructSub
 - [ ] PleIAs/SYNTH # massive
 - [x] nvidia/Nemotron-Instruction-Following-Chat-v1
 - [ ] nvidia/Nemotron-SFT-Instruction-Following-Chat-v2 # loading is broken in some way
 - [x] nvidia/Nemotron-SFT-Instruction-Following-Chat-v3
 - [x] Open-Orca/OpenOrca
 - [x] HuggingFaceH4/ultrachat_200k
 - [x] allenai/tulu-v1-sft-mixture
 - [x] allenai/tulu-v2-sft-mixture
 - [x] allenai/tulu-3-sft-mixture
 - [x] allenai/tulu-v3.1-mix-preview-4096-OLMoE
 - [x] Magpie-Align/Llama-3-Magpie-Pro-1M-v0.1
 - [x] Magpie-Align/Llama-3-Magpie-Air-3M-v0.1
 - [x] Magpie-Align/Magpie-Qwen2.5-Pro-1M-v0.1
 - [x] Magpie-Align/Magpie-Qwen2.5-Math-Pro-300K-v0.1
 - [x] Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1
 - [x] Magpie-Align/Magpie-Qwen2-Pro-200K-English
 - [x] arcee-ai/The-Tome 

### Reasoning
 - [x] facebook/natural_reasoning + instruction
 - [x] nvidia/AceReason-1.1-SFT
 - [x] nvidia/AceMath-Instruct-Training-Data

### MCQA
 - [x] nvidia/OpenScience
 - [x] nvidia/OpenScienceReasoning-2
 - [x] nvidia/Nemotron-Science-v1
 - [x] allenai/sciq

### Math
 - [x] math-ai/StackMathQA -> stackmathqa800k
 - [x] meta-math/MetaMathQA + instruction + bench-format
 - [x] TIGER-Lab/MATH-plus + instruction + bench-format
 - [ ] nvidia/OpenMathReasoning # not all correct?
 - [x] nvidia/OpenMathInstruct-1
 - [x] nvidia/OpenMathInstruct-2
 - [x] nvidia/Nemotron-PrismMath
 - [x] OpenDataArena/ODA-Math-460k
 - [ ] OpenDataArena/MathLake # big and hard to check contents

### Code
 - [x] nvidia/OpenCodeReasoning
 - [ ] nvidia/OpenCodeReasoning-2 # no questions?
 - [x] nvidia/Nemotron-Math-Proofs-v1
 - [ ] nvidia/Nemotron-Competitive-Programming-v1 # a lot of reasoning with no clear solution
 - [x] nampdn-ai/tiny-codes
 - [x] Goedel-LM/Goedel-Pset-v1

### Special
 - [ ] github.com/allenai/natural-instructions (group by task)
 - [ ] bigscience/P3 (group by task)


## Single Sequence

### Long context
 - [ ] institutional/institutional-books-1.0 (filter to english, split by gen pages?)
 - [ ] Lyun0912/LongABC (naive or semantic chunking?)
 - [ ] common-pile/project_gutenberg_filtered (naive or semantic chunking?)
 - [ ] zkeown/gutenberg-corpus (split by chapter or paragraph)
 - [ ] Despina/project_gutenberg (use existing chunks)
 - [ ] sxiong/DHSA_Long-Data-Collections (naive or semantic chunking?)

### Knowledge
 - [ ] HuggingFaceTB/smollm-corpus/cosmopedia-v2 (group by format, semantic cluster)
 - [ ] HuggingFaceTB/smollm-corpus/fineweb-edu-dedup (semantic cluster)
 - [ ] PleIAs/SYNTH (filter english, group by url)

### Code
 - [ ] Avelina/python-edu-cleaned (semantic cluster)
 - [ ] codeparrot/github-code (group by repo)
 - [ ] project-themis/git-commits (group by repo)
 - [ ] bigcode/commitpack (group by repo)
 - [ ] Ujjwal-Tyagi/notabug (group by repo)
 - [ ] bigcode/jupyter-parsed (group by repo)

## Hidden Variable

 - [ ] RyokoAI/Fandom23K (group by tag, semantic cluster half of groups)
 - [ ] MLNTeam-Unical/PersonaGen (group by persona and model)
 - [ ] PygmalionAI/PIPPA (semantic cluster by character)
 - [ ] Asap7772/persona_paired (group by persona)
 - [ ] PaDaS-Lab/webfaq + PaDaS-Lab/webfaq-v2 (filter english, group by origin)
 - [ ] barilan/blog_authorship_corpus (group by author)
 - [ ] HuggingFaceTB/stackexchange_2025_md (group by answer user id)
 - [ ] vngclinh/goodreads-reviews (group by user id)
 - [ ] McAuley-Lab/Amazon-Reviews-2023 (group by user id)
 - [ ] Johnnyeee/Yelpdata_663 (group by user id) 
 - [ ] corbt/enron-emails (group by sender, sort by date)
 - [ ] gfissore/arxiv-abstracts-2021 (group by author/submitter)
 - [ ] code-search-net/code_search_net (group by repo/file, prompt with comment?)
 - [ ] PaDaS-Lab/moltbook-corpus (group by author/user id)
 - [ ] open-index/hacker-news (group by author/user id)
 - [ ] LxYxvv/quora_qa_raw (group by answer author id)
 - [ ] xywang1/OpenCharacter/Synthetic-Dialogue (group by character_id)
 - [ ] LongLaMP/LongLaMP (group by author)
 - [ ] bowen-upenn/PersonaMem-v2
 - [ ] Neph0s/CoSER (group by novel/character)
 - [ ] cornell-movie-dialog/cornell_movie_dialog (group by movie/characters)
 - [ ] webis/tldr-17 (group by author)
 - [ ] Kaylee0501/ImplexConv-opposed + Kaylee0501/ImplexConv-supportive (group by conversation)
 - [ ] arranonymsub/HiCUPID (group by user)
 - [ ] common-pile/caselaw_access_project (filter generic, group by author)
 - [ ] blitt/SPoRC (group by podcast, speaker)
 - [ ] open-index/arctic (group by author)
 - [ ] gk4u/reddit_dataset_139 (group by user)
 - [ ] biglam/hansard_speech (group by speaker id)
 - [ ] AAU-NLP/HiFi-KPI (group by company)
 - [ ] bigcode/starcoder2data-extras/issues (group by repo)
 - [ ] kaggle.com/datasets/kaggle/meta-kaggle-code (group by user or competition)
 - [ ] mabujadallah/GitHub-Agentic-PR-Dataset (group by repo or user)
 - [ ] ShayManor/Labeled-arXiv (group by author)
 - [ ] ray0rf1re/AO3-2020 (group by story id)
 - [ ] ExponentialScience/DLT-Patents (group by inventor, assignee)
 - [ ] alpindale/two-million-bluesky-posts (group by author)
 - [ ] AmaanP314/youtube-comment-sentiment (group by video or author)
 - [ ] clips/mqa (group by domain)

## Tools

- [ ] Agent-Ark/Toucan-1.5M
