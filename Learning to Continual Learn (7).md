# Main

![][image1]

# Learning to Continual Learn

# Introduction

## Overview

Continual learning is when models continue to learn and improve during deployment. Good continual learning should have the following properties:

1. High sample efficiency  
2. Low forgetting of known information  
3. Learn from interactions and feedback/rewards  
4. Keep learning for a long time  
5. Deep skill acquisition

LLMs can already perform loose continual learning through in-context learning. This includes both [learning from examples](https://arxiv.org/abs/2005.14165) and learning from rewards (reinforcement learning) \[[1](https://arxiv.org/abs/2410.05362)\] \[[2](https://arxiv.org/abs/2410.05362v1)\]. However, in-context learning struggles with the deep skill acquisition and strong changes to model behavior that is expected from continual learning.

## The Zeitgist

Many experts, including [a16z](https://a16z.com/why-we-need-continual-learning/), [Ilya Sutskever](https://www.dwarkesh.com/p/ilya-sutskever-2), [Richard Sutton](https://www.dwarkesh.com/p/richard-sutton), and [Demis Hassabis](https://www.bigtechnology.com/p/google-deepmind-ceo-demis-hassabis-946) have singled out continual learning as one of, if not *the* biggest open problem in AI research.  The biggest missing pieces right now are sample efficiency (generalization) and forgetting.

Current startups in the continual learning field fall into a few categories:

### 1\. Harnesses & Retrieval

These startups are betting that by managing text-based context, external retrieval and harnesses and can turn LLMs into continual learners without modification. The advantage of this direction is that it is compatible with existing LLM APIs, and doesn’t require any training.

**Examples:** [Letta](https://www.letta.com), [mem0](https://mem0.ai/), [Supermemory](https://supermemory.ai/), [xmemory](https://xmemory.ai/)

### 2\. Data

These startups are betting that with a high enough quantity or quality of data, post-training methods like RL and SFT can turn LLMs into continual learners. The advantage of this direction is that new models don’t need to be trained from scratch or modified with up-front training, only post-trained.

**Examples:** [Applied Compute](https://www.appliedcompute.com/), [Trajectory](https://trajectory.ai/), [Metis](https://www.withmetis.ai/)

### 3\. Post-Training Adaption

These startups are betting that methods can be developed to make post-training modifications to pretrained models to enable continual learning. The advantage of this direction is that new models don’t need to be trained from scratch or up-front, only modified on demand.

**Examples:** [Adaption](https://www.adaptionlabs.ai/), Sublinear Systems, [Sakana](https://sakana.ai/) (kinda)

### 4\. Architectures

These startups are betting that new non-transformer architectures will enable continual learning. The advantage of this direction is that directly training models to be continual learners creates research flexibility and opens up new possibilities.

**Examples:** [Core Automation](https://www.coreauto.com/#approach), [Magic](https://magic.dev), Learning Machine

### Wildcards

These startups either lack information or don’t fit into the above categories.

**Examples:** [Flapping Airplanes](https://flappingairplanes.com/), [SSI](https://ssi.inc/), [Ineffable Intelligence](https://www.ineffable.ai/)

# Vision

We stand to differentiate ourselves from other players with our unique technical “bet” on the future of continual learning and AI in general. Nobody knows what the actual answer will look like, so funders want to diversify against all future possibilities. By investing in us, funders can ensure that they win if our bet turns out to be the correct one.

## Our Bet

*“Learning to continual learn will eventually dominate all other continual learning approaches.”*

We are planting our flag on the idea that outer-loop optimization applied to continual learning will eventually surpass and dominate other continual learning approaches.

This is what you might call the “[Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) [pilled](https://x.com/karpathy/status/1973435013875314729?lang=en)” (and the older [The Unreasonable Effectiveness of Data](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/35179.pdf)) approach to continual learning. Humans, for all of our strengths, can only conceive of strategies with a certain level of complexity. Given sufficient optimization resources, a general optimization algorithm over a large search space can almost always find better solutions than us. We think this is also true for continual learning algorithms. 

At the start, our approach will not look too different from the [sequence model based continual learning](https://arxiv.org/abs/1611.02779) that has existed for a decade+. However, by combining modern scale with architectural and algorithmic improvements, we believe that we can fulfill the vision that previous work just couldn’t reach (as is often the case in the history of AI).

How does this solve the specific problems that currently plague continual learning?

1. **Sample efficiency.** This will be our primary objective. It has already been shown that LLMs already exhibit [emergent learning abilities](https://arxiv.org/abs/2005.14165), and that the sample efficiency of these abilities can be improved through [meta-learning](https://arxiv.org/abs/2110.15943). We believe that by focussing directly on this, with better methods and larger scale, we can achieve much higher sample efficiency than transformer ICL (which is [about the best we have right now](https://arxiv.org/pdf/2110.15943)).

2. **Catastrophic Forgetting.** As has been [discussed at length](https://arxiv.org/abs/2312.00752) in the sequence-modelling literature, training through a compressed state can teach the model how to effectively overwrite old information without forgetting what’s important. We believe that with large states and extremely long training horizons, catastrophic forgetting will get optimized away.

To further support our claim with some crackpot logic, we point to the fact that outer-loop optimization has already built a continual learning model at least once: *humans.* Evolution is nature’s ultimate optimization loop. It operates over an unimaginably large search space, and more relevantly, it operates at the most outer level of the design loop. It captures each organism’s entire life, every event and task from birth to death, in a single sample of the fitness function. This contrasts with traditional pretraining that optimizes each “event” (conversation in the case of LLMs) independently, failing to recognize inter-episode interactions. Our goal is to replicate this holistic optimization *in silico*: wrapping entire multi-episode (or long-horizon) deployments into single samples, and optimizing them together.

## Secondary Technical Theses

### Extrapolation

Continual learning (and sequence-modelling) architectures should have semi-guarantees about behavior and convergence beyond the horizon length that they were (meta-)trained on. This is because deployment horizons will quickly eclipse all but the best training data. This will also allow us to efficiently bootstrap on-policy training to longer and longer horizon lengths, since we can incrementally increase our horizon lengths without worrying about catastrophic failure (which is basically an unrecoverable state for on-policy learning). This overcomes the exact reason why [some claim that you can’t imitation learn how to continual learn](https://www.lesswrong.com/posts/9rCTjbJpZB4KzqhiQ/you-can-t-imitation-learn-how-to-continual-learn).

### Reward Models

Test-time scaling [is critical for the performance of modern AI systems](https://www.tobyord.com/writing/mostly-inference-scaling). These scaling strategies often rely on a reward model or verifier in addition to the obvious generator. In fact, [the verifier is often the bottleneck](https://www.tobyord.com/writing/mostly-inference-scaling) in high scaling regimes. This implies that if we want our continual learner to have test-time scaling abilities, the verifier should also continually improve.

In situations where the model is not getting frequent feedback from the environment, strong reward models could also be important for adapting the policy using things like [self rewards](https://arxiv.org/abs/2401.10020).

### Free Energy

Standard RL is [very information inefficient](https://www.tobyord.com/writing/inefficiency-of-reinforcement-learning), so [we probably need something else](https://x.com/karpathy/status/1921368644069765486). Our belief is that [free energy minimization](https://en.wikipedia.org/wiki/Free_energy_principle) , which is related to the buzzword “[world models](https://techcrunch.com/2026/03/09/yann-lecuns-ami-labs-raises-1-03-billion-to-build-world-models/)”, is the answer *when combined with meta-learned continual learning*. In our context, free energy minimization is training the model to predict inputs, observations, and feedback in addition to optimizing its actions. To see why this may help, suppose that we have a model operating in a typical reinforcement learning loop: *a\_t\~\\pi\_\\theta(s\_t)*; *r\_t,s\_{t+1}=f(s\_t, a\_t)*. Next, suppose that we meta-train a model on multi-episode horizons on a variety of environments, and during each trajectory the model is trained using both an RL loss and a free energy loss of the form *d(f\_\\theta(s\_t, a\_t), r\_t, s\_{t+1})* on each episode. This will induce updates to the model’s weights: *\\Delta \\theta\_{RL}* from the RL loss, and *\\Delta \\theta\_{FE}* (which we can expect to be [much more information-dense](https://youtu.be/Ount2Y4qxQo?si=XCN68c3rV8Lq08hg&t=1165)) from the free energy loss. Then, the model will be trained with RL and FE objectives on later trajectories, *conditioned on* *\\Delta \\theta\_{RL} and \\Delta \\theta\_{FE}*. The key is that during meta-learning, the model will learn to use the *\\Delta \\theta\_{FE}* state to inform future action decisions, and that the high information density will lead to higher sample efficiency than standard RL. Why would the model do that? Well, because [models just want to learn](https://vrungta.substack.com/p/models-just-want-to-learn-how-generative), and they have already been shown to [make use of sub-optimal state updates](https://arxiv.org/abs/1609.01596). If necessary, we could further encourage it through [end-to-end backpropagation](https://arxiv.org/abs/2512.23675), but that would be more expensive.

If the model-free version doesn’t work directly, then shifting towards explicit model-based RL where the state transition model continually learns would require fewer assumptions. This would yield a search and planning test-time scaling loop.

*Edit:* [This paper](https://arxiv.org/abs/2605.24517) on adding predictive auxiliary losses to agentic RL just came out, and is strong evidence supporting this thesis. [This recent paper](https://arxiv.org/pdf/2605.28548) provides less direct evidence.

### Priors (weak)

The real world is messy and [full of distinct priors](https://arxiv.org/abs/2511.14761) compared to the range of possible universes. This means that the [Solomonoff priors](https://en.wikipedia.org/wiki/Solomonoff%27s_theory_of_inductive_inference) that [some startups](https://ndea.com) are betting on will not translate well to the real world (though they probably will work well in white rooms like math and code). Learned priors, which a meta-learner will discover, will work better in the majority of situations where humans are typically expected to act.

### Synthetic Data (weak)

In its most general form, the [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) in which reinforcement learners act is composed of only a few components: the state space, the action space, the state transition function, and the reward function. The more restrictive [bandit setting](https://en.wikipedia.org/wiki/Multi-armed_bandit) (which contains a lot of useful behavior including the majority of traditional LLM benchmarks) removes the state transition function, leaving us with only a state space, action space, and reward function. Our theory is that by procedurally generating reward functions (such as through system prompts or latent variables) and keeping the state and action spaces fixed (being just text for language models), we can create a scalable meta-RL training ground. Through multiple interactions with a given hidden reward function, the meta-learner will figure out how to minimize the regret on new hidden reward functions (which is what a task really is).

## Guiding Principles

### Architecture Retrofitting

A startup with less than 8 or 9 figures of funding should never train foundation models from scratch. It is much cheaper, faster, and likely more effective to start with the models that larger players have already built. For this reason, research on novel architectures should prioritize designs that are compatible with existing models. To be compatible, novel architectures should be non-destructive during conversion and leverage the existing circuitry in models rather than attempting to forge entirely new computation paths.

### Backwards Compatibility

The users of a continual learning platform will have significant investment and reliance on the skills and knowledge baked into their custom model instance. At the same time, the provider will be releasing new models with better capabilities. A high-value (but very unsolved) research problem will be making new model releases backwards compatible with the learning done by previous releases, so that users can migrate to the new release without friction or alienation.

# Business Plan

## Short-Term

## ![][image2]

Our first release will be a (possibly closed beta) continually learning API (probably a flash model tuned for tool-calling) that gets better the more you use it. This assumes that the model gets some kind of feedback/rewards, but we might also see slight improvement just using something like [self rewards](https://arxiv.org/abs/2401.10020) (or only free energy, see above). To enable feedback/rewards, it may make sense to partner with one of the many companies building platforms to enable that (or use an open-source one). 

The keys to differentiation and success are:

1. **High ability at low cost.** We probably won’t have enough resources to build a model that competes with frontier ability out-of-the-box (or even after adaptation in some cases). Instead, our adapted models will provide the best ability relative to cost and speed. This is a valuable and [very possible](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/) position on the [Pareto](https://en.wikipedia.org/wiki/Pareto_front) [frontier](https://www.reddit.com/r/LocalLLaMA/s/IpYs7Sj7Hk), [especially in agentic settings](https://arxiv.org/abs/2506.02153).

2. **High sample efficiency.** There are already a slew of companies providing RL and fine tuning as-a-service. Our key differentiator and advantage over these competitors will be sample efficiency. Our meta-learned models will require fewer examples. This will enable higher performance with less investment, and unlock new use-cases that would otherwise be impossibly sample intensive. We also hope to unlock advantages in multi-task learning and forgetting minimization.

3. **Ease-of-use.** Training and inference are currently separate workflows. By unifying them under a single API with minimal inference and feedback functions, we create a single simple workflow. 

*Edit:*

Another good first product could be a small (we call it “efficient”) coding agent model that learns through “editing”. The workflow would be:

1. The user makes a coding request.  
2. The agent implements changes, and maybe runs tests.  
3. The user reviews the diff and test results.  
4. The user makes changes to the diff and/or code.  
5. The model is updated using either SFT loss or preference loss to make it more likely to produce the human’s diff. We could also include [ECHO](https://arxiv.org/abs/2605.24517)\-like “free energy” auxiliary losses for test results and/or written feedback.  
6. [TTT-Discover](https://arxiv.org/abs/2601.16175) optimization loops could be implemented using  just rejection sampling.

This would be particularly easy to build because of the massive amount of coding agent data that exists and continues to be produced. It is also a relatively easy field to perform on-policy reinforcement learning. 

In the short-term, we would be stuck with a relatively small mode. In the more medium-term, it would not take too much scaling to get to [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (which some people actually use).

Continual learning coding data is also readily available at scale using timelines of changes within single repos as horizons. This would provide only SFT training, but that’s all you need for the diff workflow described above. Our results on out-of-distribution few-shot supervised learning show the viability of our methods to make this work with unique sample efficiency.

## Medium-Term

![][image3]

Our more mature release will be a continually learning API or product that improves the more you use it.

With more funding, we should be able to build a stronger model (still retrofitted to an open-source base) that actually exceeds the ability of frontier models after enough specialized adaptation. With that in mind, we focus on serving researchers/engineers whose problems are difficult to solve for general models, but our specialized/experienced models might be able to tackle. That demographic is smaller but higher paying, and you avoid competing in the economy-of-scale race to the bottom (and it's more fun).

Rather than fitting models to individual users, it makes more sense to aggregate feedback and learning from teams and communities into a single model to maximize the quantity of feedback and rate of learning.

A good example of a narrow product would be a continual learning copilot/autocomplete/next-edit-prediction tool, since they are constantly getting feedback from acceptances and rejections (though this bets against the trend of hands-off agentic work).

Another high-impact application of these models could be [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)\-style optimization loops for climbable objectives. [TTT-Discover](https://arxiv.org/abs/2601.16175) has already demonstrated reinforcement learning (weight-space optimization) as a strong alternative to text-space optimization. Our models that have been meta-trained to perform narrow RL should offer improved efficiency and performance over the already formidable TTT-Discover framework.

## Long-Term/Moonshot

![][image4]

The graph says it all.

We justify this possibility with an appeal to authority. As mentioned in the introduction, experts like Ilya Sutskever and Richard Sutton believe that continual learning will unlock both new applications and stronger long-term abilities.

# Questions

## Why us?

The AI industry is flooded with both large and small labs with goals similar to ours. This section describes  how we compete with the incumbents.

1. **A concrete design.** Our design for our continual learning architecture is more concrete and actionable than any openly known alternatives. This will allow us to reach a beta or release quickly (and just doing that might be enough for acquisition/aqui-hire). Being first is a big deal for continual learning because of the lock-in effect: when you switch to a different model, you need to restart teaching it (see backwards compatibility thesis).

2. **Outsider team.** Many have criticized the large labs for their over-exploitation of the GPT/LLM paradigm and lack of exploration of new ideas. In what may be a new [“age of research”](https://www.dwarkesh.com/p/ilya-sutskever-2), fresh minds and new ideas are an advantage. 

3. **Building efficiently.** It has been mentioned several times now, but retrofitting existing models with a new architecture instead of training from scratch is huge for a small startup. Other labs that train from scratch (either due to architectural incompatibility or reputational stakes) will be burning much more time and money building their models.

## Why now?

* Previous meta-RL experiments had to rely on bespoke or limited data. Now, the emergence of agentic AI has caused a massive influx in the creation of reinforcement learning datasets and environments. This means that there is an emerging opportunity to perform modern large-scale meta-RL pretraining (and scaling previously niche methods to large scale is often the source of large AI breakthroughs). 

# Evidence

## What we have

### Sequence Modelling

This is Phase 1 of the technical plan.

Here we treat long-context sequence modelling as a meta-learning problem. Each “example” is a chunk of the sequence (overlapping: 0-1024, 0-2048, 1024-3072…) in the style of [iTTT](https://aklein.bearblog.dev/ittt/). 

In this experiment, we started with the pretrained [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model and trained it for 2 billion tokens on the [Lyun0912/LongABC](https://huggingface.co/datasets/Lyun0912/LongABC) dataset. The runs are as follows:

1. **baseline:** Each chunk is trained and evaluated independently. Standard Llama architecture.  
2. **no-meta:** The same checkpoint as **baseline**. Fast weights are sequentially updated like **meta** at evaluation time, but not during training (new parameters are freshly initialized).  
3. **meta:** A meta-trained model where fast-weights are sequentially updated after each chunk at both training and evaluation time.

| ![][image5] | ![][image6] |
| :---- | :---- |

A notable result from this experiment is that **baseline** and **no-meta** have essentially the same curve. We had previously found that, before fine-tuning on the dataset (so the original Llama weights), fast weights with no meta-training could significantly improve over the baseline. The fact that there is no improvement after fine-tuning implies that the untrained fast weights were only assisting the model with adapting to the new data domain, which is not needed after fine-tuning. The fact that the meta-trained fast weights do significantly improve performance shows the effectiveness of our architecture.

### Few-shot Learning

To test models’ abilities to learn from a few supervised examples, we used the [ManyICLBench](https://huggingface.co/datasets/launch/ManyICLBench/viewer/clinc150?row=0) benchmark. This benchmark presents a context-length-based set of examples from a specific task, and evaluates the model’s accuracy on held-out examples.

Our evaluation included the following details:

1. We kept only the “easy to evaluate” tasks from the benchmark. This meant removing the math, translation, and chain-of-thought tasks.  
2. Only 10 held-out tasks are evaluated for each seed. This is due to a limitation in our evaluation harness: the model can see previous held-out tasks during later ones, significantly warping the results at supposedly low context lengths. However, the small sample size is mitigated by the 3 different seeds per task and context length.  
3. Examples are only counted as correct if the model’s argmax logits are equal to the answer’s tokens at every position.

The methods tested are as follows:

1. **OLoop (ours):** Our meta-trained model. We used the same checkpoint as for the **meta** run in the sequence modelling section above. Updates were performed using the same [iTTT](https://aklein.bearblog.dev/ittt/)\-like inference algorithm after concatenating each task’s examples into a sequence.  
2. **LoRA:** The original [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model with LoRA adaptation. Similar to **OLoop**, we update the weights after each overlapping chunk [iTTT](https://aklein.bearblog.dev/ittt/)\-style. To make this a fair and formidable baseline, we used [PiSSA](https://proceedings.neurips.cc/paper_files/paper/2024/file/db36f4d603cc9e3a2a5e10b93e6428f2-Paper-Conference.pdf) initialization, the Muon inner optimizer, placed LoRA adapters on every non-embedding matrix, and matched the **OLoop** fast-weight size (rank 96 \-\> 4M parameters per layer)**.** Some hyper-parameter tuning (learning rate, momentum) could help, but we used the same lr=1e-3 and beta=0.75 that gave the best results for the **no-meta** run in the sequence modelling section above.  
3. **ICL:** Standard in-context learning with [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) by concatenating examples the same way as in **OLoop**. Note that the model has a trained context length of 128K (and some “128K“ sequences may slightly exceed this).  
4. **Sliding:** The same as ICL but with overlapping sliding-window attention (equivalent to **LoRA** with learning rate 0).

| ![][image7] | ![][image8] |
| :---- | :---- |

We see that our model outperforms both in-context learning and fine-tuning at every context length greater than 4k. In fact, **OLoop is 3.25x more sample efficient than LoRA** (piecewise linear interpolation: 40k **Oloop** \~ 128K **LoRA**).

This result is particularly notable because:

1. **OLoop** was trained on only 2B tokens (100-120 H100 hours).  
2. The training data for **OLoop** was not designed for meta-learning, it was [general long-context text](https://github.com/Lyun0912-wu/LongAttn). Task-style and in-context learning is out-of-distribution for the new parameters. [MetaICL](https://arxiv.org/abs/2110.15943) suggests that there are more gains to be had just by training on in-distribution data.  
3. The maximum meta-training context length for **Oloop** was 32K. In spite of that, **Oloop** continues to show gains at 128K context lengths (4x extrapolation).  
4. Unlike attention, the **Oloop** architecture scales with the desirable O(1) memory and O(n) compute at inference time, while also having uniquely low O(1) memory and O(n) compute training requirements.

The implication of this result (if it holds under further scrutiny) is that commercial viability could be closer than we think. If we can push the gains over ICL and LoRA further, to say a 5x increase in sample efficiency, then we could put out a “supervised finetuning-as-a-service but with 5x less data” API or platform (more discussion can be found in the short-term business plan section above). This would then provide a stepping-stone to more powerful feedback-based and reward-based learning research.

## Demo

Note that time estimates from now are working days from what we have as of May 19th, including needing to meta-train.

### 1\. SVG Images

The model continually learns to produce SVG images. We record the progress over a horizon and present it visually.

We could present the progress as a grid/list of images or an animated gif. We could include a non meta-learned model side-by-side for comparison. We could also show a graph of rewards over time.

The reward function is based on the rendered SVG image, and could use an LLM judge or a model like [ImageReward](https://arxiv.org/abs/2304.05977). We could iteratively optimize for single prompts (a pelican on a bicycle?) in the style of combinatorial optimization, SVGs in general, or something in the middle like a semantic cluster of prompts. We could also optimize for similarity (like CLIP score) with a reference image, to give a clear target and reduce the entropy of model generations. We could even optimize for something weird like JPEG compressibility/incompressibility.

Small models can be finetuned (and use constrained decoding) to produce SVG images [relatively easily](https://huggingface.co/datasets/OmniSVG/MMSVG-Illustration). We can also include some (synthetically rewarded) offline SVG data during meta-learning. We could even include a specialized on-policy meta-RL training stage specifically for SVGs. There’s a chance that our fully optimized image(s) actually look better than some frontier models and we could boast a comparison.

The idea is that this is a toy version of the reward function feedback loops that one might see in deployment. Those loops could be [iterative optimization on the same input](https://arxiv.org/abs/2601.16175) or learning a more general domain. Real-world reward functions could include:

* Verifiable rewards (math/code [optimization](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) or correctness)  
* Human feedback  
* Reward model or LLM-as-a-judge  
* Auto-research AI experiments  
* Physics simulators or experimental results

We might get asked by a more technical person about how the model could incorporate more informative feedback (logs, written human feedback) than just rewards, and the answer is the free energy representation learning idea (see above).

**Timeline:** Must include SVG training data and implement the render-verify-improve loop. \~1-1.5 weeks of locked-in work.

### 2\. Combinatorial Optimization

The model continually learns to solve combinatorial optimization problems. We record the progress over a horizon and present it visually.

The task could be code optimization, something scientific, or something simple (traveling salesman problem). The constraint is that the model must be somewhat able to produce the correct format and get non-zero rewards at the start of the horizon. Similar to SVG images we could plan ahead and include training data.

Depending on the task, we may be able to present the optimization visually (traveling salesman path, folded protein, etc).

See the SVG image idea for more discussion on this type of demo.

**Timeline:** Depends on the task. Must set up the verification/learning loop, and make sure that the model can follow that format and get non-zero rewards (potentially from training data). We might also want custom visualizations. \~1-2 weeks of aggressive work.

### 3\. Tool Use

The model continually learns from reward feedback to use a set of tools.

The challenge is that we need many tasks (that we can verify the correctness of) for the same suite of tools. [LifelongAgentBench](https://github.com/caixd-220529/LifelongAgentBench) could be a good solution, while also being a standard continual learning benchmark.

In the real world, checks and rules implemented in the harness or platform would provide the rewards. Human feedback could also be used.

**Timeline:** Must include formatted tool data, and set up the (LifelineAgentBench?) environment/reward/learning loop. \~1 week of aggressive work.

### 4\. Verifiable Rewards

The model continually learns from verifiable rewards to solve math or code problems.

We could focus on continually learning a specific semantic cluster or domain of problem, or just math/code in general.

**Timeline:** Must implement the verification/learning loop. \~1 week of aggressive work.

### 5\. LLM-as-a-Judge

The model continually learns from reward feedback by an LLM judge.

Prompts could just come from (single-tern) chat datasets or benchmarks.

This could be advertised as a proxy for human feedback. We could also give a specific custom prompt or rubric to the reward model to show that we can continually specialize and not just optimize general goodness.

**Timeline:** Must implement the reward/learning loop. \~1 week of work.

### 6\. Few-shot Learning

The model uses supervised in-context-like few-shot learning to continually learn a task.

Could use [OmniBenchmark-1K](https://arxiv.org/abs/2602.03473) or [ManyICLBench](https://huggingface.co/datasets/launch/ManyICLBench) as benchmarks.

**Timeline:** Must meta-train an SFT model and implement the benchmarks. \~1-2 weeks of aggressive work.

### 7\. Red-teaming

The model continually learns to jail-break another LLM.

This is similar in spirit to verifiable rewards except jail-breaking success is the reward.

There are probably benchmarks for this.

**Timeline:** Must implement the reward/learning loop. \~1 week of aggressive work.

### 8\. Game Playing

The model continually learns to play a game.

Games could include obvious ones like chess and checkers, or something else.

Rewards could come from self-play or from playing against an automated baseline.

The constraint is that the model must be somewhat able to produce the correct format and get non-zero rewards at the start of the horizon. The back-and-forth nature of many games could also cause difficulty.

**Timeline:** Depends on the game. Unknown amount of work.

# Technical

# Methods

## Literature

### Sequence Modelling

* [TTT-E2E](https://arxiv.org/abs/2512.23675): Unifies sequence modelling and continual learning. This is the only published sequence modelling architecture that fulfills our extrapolation requirement, *but nobody else seems to have noticed that.*  
* [In-Place TTT](https://openreview.net/forum?id=dTWfCLSoyl): A less thorough architecture for continual learning as sequence modelling. Targets likelihood updates but does not enforce them strictly (and therefore doesn’t necessarily fit our extrapolation requirement.  
* [Nested Learning](https://arxiv.org/abs/2512.24695): Brought attention to the concept of nested optimization loops, but the Hope architecture is really still just a KV-binding TTT sequence modelling architecture.  
* [MesaNet](https://arxiv.org/pdf/2506.05233): A sequence modelling architecture that iteratively solves the entire prefixed mesa-optimization objective (usually L2 loss) at each token/chunk. Basically the ultimate instantiation of the connection between linear attention and mesa-optimization (an ICL mechanism, see below). Competitive but very slow.  
* [PERK](https://arxiv.org/html/2507.06415v1): Learn to perform long-context modelling by meta-learning LoRA parameters. A less advanced version of iTTT, and a precursor to TTT-E2E. Uses truncated gradient unrolling, an encoder-decoder like loss (rather than fully autoregressive), per-layer per-step learning rates calculated using high-order gradients, and the AdamW inner-loop optimizer.  
* [Dynamic Eval](https://proceedings.mlr.press/v80/krause18a): An early ancestor of iTTT that performs chunk-wise gradient descent. Not meta-learned.  
* [FWL](https://arxiv.org/pdf/2212.02475): Another ancestor of iTTT that meta-learns on top of chunk-wise gradient descent, but computes gradients in parallel.

### Test-Time Training

* [Effective TTT](https://arxiv.org/abs/2411.07279v1): Taking gradient steps on supervised few-shot examples greatly boosts few-shot learning performance.  
* [TTT Provably Improves](https://openreview.net/forum?id=bma2FB5MNs): Theory behind TTT and why it works.  
* [TTT-NN](https://openreview.net/forum?id=CNL2bku4ra): Taking supervised gradient steps on retrieved nearest neighbors improves performance.  
* [TTT-SIFT](https://arxiv.org/abs/2410.08020): An improved curriculum/data-selection method over NN.  
* [TTC-RL](https://arxiv.org/abs/2510.04786): Taking RL gradient steps on retrieved similar examples boosts performance (basically an RL version of TTT-SIFT).  
* [GC-TTT](https://arxiv.org/abs/2507.18809): Taking offline (compared to online in TTC-RL) RL gradient steps on retrieved similar examples boosts performance in goal-conditioned RL.

### Test-Time Scaling

* [NCO](https://arxiv.org/abs/1611.09940): Early work on reinforcement learning as a search algorithm for combinatorial optimization.  
* [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/): The canonical example of LLM-guided text-space evolution for optimizing climbable objectives.  
* [MIGRATE](https://arxiv.org/pdf/2508.08641): GRPO RL on LLMs as a test-time search algorithm for difficult climbable objectives.  
* [ThetaEvolve](https://arxiv.org/abs/2511.23473): Another example of AlphaEvolve-like optimization using LLMs and test-time RL.  
* [TTT-Discover](https://arxiv.org/abs/2601.16175): Improves over ThetaEvolve. Currently state-of-the-art.  
* [Algorithm Discovery](https://arxiv.org/abs/2504.05108): Another example of LLM-based test-time RL as a search algorithm for optimizing algorithms.  
* [EvoX](https://arxiv.org/abs/2602.23413): Meta-learning of evolutionary search/optimization algorithms (not RL).

### In-Context Learning

* [GPT-3](https://arxiv.org/abs/2005.14165): The discovery of emergent few-shot/in-context learning in transformers.  
* [MetaICL](https://arxiv.org/abs/2110.15943) / [MetaICT](https://aclanthology.org/2022.acl-long.53.pdf): Meta-learning to improve ICL performance.  
* [Many-Shot ICL](https://arxiv.org/abs/2404.11018): Scaling ICL to many in-context examples.  
* [ICRL](https://arxiv.org/abs/2410.05362v1): LLMs can perform in-context RL, but naive methods fail.  
* [Bandit ICRL](https://arxiv.org/abs/2410.05362): LLMs can perform in-context bandit RL.  
* [Induction Heads](https://arxiv.org/abs/2209.11895): Induction heads as a source of ICL performance.  
* [Mesa-Opt](https://arxiv.org/abs/2309.05858): Mostly theoretical discussions of mesa-optimization mechanisms in transformers and their role in ICL.

### Continual Learning

* [MemRL](https://arxiv.org/pdf/2601.03192): Non-parametric runtime reinforcement learning for agents. Basically estimates and updates Q-values in-context.  
* [CASCADE](https://arxiv.org/abs/2605.06702): Formalizes deployment-time learning, providing examples, benchmarks, and a baseline in-context method.  
* [CaRE](https://arxiv.org/abs/2602.03473): Multi-task continual learning with low forgetting using sparse MoEs, includes benchmarks and baselines for learning up to 300 tasks.  
* [LifelongAgentBench](https://arxiv.org/pdf/2505.11942): An interactive benchmark for lifelong learning in agents.  
* [FST](https://arxiv.org/pdf/2605.12484): Fast-Slow training optimizes fast text-space context in addition to typical slow-weight updates. Improves sample efficiency and forgetting. Not meta-learned. 

### Meta-Learning

* [MAML](https://arxiv.org/abs/1703.03400): The base of much modern meta-learning. FOMAML is particularly relevant to our methods.  
* [Reptile](https://arxiv.org/abs/1803.02999): Foundational theory on first-order meta-learning algorithms.  
* [Universal ML](https://arxiv.org/pdf/1710.11622): Theory and conditions for universal meta-learning.  
* [P-MAML](https://arxiv.org/abs/1806.02817) / [U-MAML](https://arxiv.org/abs/1907.11864): Probabilistic parameters in MAML for uncertainty, task sampling, and bayesian inference.  
* [CAVIA](https://arxiv.org/abs/1810.03642) / [BOIL](https://arxiv.org/abs/2008.08882): Variants of MAML where only a subset of parameters are fast and the rest are slow (like our methods).  
* [Meta-SGD](https://arxiv.org/pdf/1707.09835): A generalization of MAML that learns per-parameter learning rates.   
* [La-MAML](https://proceedings.neurips.cc/paper_files/paper/2020/file/85b9a5ac91cd629bd3afe396ec07270a-Paper.pdf): MAML with per-parameter learning rates (learned through meta-hypergradients like iTTT) and look-ahead.  
* [MAML-en-LLM](https://arxiv.org/abs/2405.11446): Applying MAML to LLMs.  
* [Text-to-LoRA](https://arxiv.org/abs/2506.06105): Hypernetworks can be meta-trained to produce LoRA adapters for tasks given their description.  
* [Doc-to-LoRA](https://arxiv.org/abs/2602.15902): Hypernetworks can convert context into equivalent LoRA adapters for LLMs.  
* [LiFT](https://openreview.net/forum?id=7nyJBVCTGQ) / [ABMLL](https://arxiv.org/abs/2508.14285v2): Bayesian meta-learning over LoRA adapters.  
* [Task Augmentation](https://proceedings.mlr.press/v139/yao21b/yao21b.pdf): Data augmentation by modifying the task function (such as by shuffling label indices) increases meta-learning performance.  
* [FO-BLO](https://arxiv.org/pdf/2006.03631): Unbiasing FOMAML-style training.  
* [Task Diversity](https://arxiv.org/abs/2208.01545): The importance and impact of non-hemogenous tasks in meta-learning.

### Meta-RL

* [RL^2](https://arxiv.org/abs/1611.02779): Early work on meta-RL using RNNs.  
* [MACAW](https://arxiv.org/abs/2008.06043): MAML-like offline meta-RL.  
* [Algorithm Distillation](https://arxiv.org/abs/2210.14215): In-context meta-RL with transformers.  
* [Scaling AD](https://arxiv.org/pdf/2506.13892v1): Algorithm Distillation with Mamba for longer contexts.  
* [Meta-DT](https://proceedings.neurips.cc/paper_files/paper/2024/file/4f3820576130a8f796ddbf204c841487-Paper-Conference.pdf): In-context offline meta-RL with transformers.  
* [OPEN](https://arxiv.org/pdf/2407.07082): Meta-learned optimizers for RL.   
* [IDAQ](https://arxiv.org/pdf/2305.19529): Addresses the distribution shift between off-policy training and on-policy sampling in meta-RL   
* [MR-Search](https://arxiv.org/abs/2603.11327): Learning to in-context reinforcement learn with self-reflection for agentic search.  
* [Prefrontal cortex](https://deepmind.google/blog/prefrontal-cortex-as-a-meta-reinforcement-learning-system/): Parallels between meta-reinforcement learning and the prefrontal cortex. Empirical correlations between artificial neural networks trained using meta-RL and psychology experiments.

### Off-Policy RL

* [AWR](https://arxiv.org/pdf/1910.00177): A simple method for off-policy RL with LLMs.  
* [DRO](https://arxiv.org/abs/2405.19107): Probably the best method for off-policy LLM RL in the bandit setting. Requires a learnable V function that we would rather go without.  
* [DQO](https://arxiv.org/pdf/2410.09302) / [OREO](https://aclanthology.org/2025.findings-acl.464.pdf): Generalization of DRO to the non-bandit setting.  
* [A\*-PO](https://arxiv.org/abs/2505.20686): An off-policy but not fully offline LLM RL algorithm with a close connection to DRO: uses an empirical V instead of a learned one.  
* [CQL](https://arxiv.org/abs/2006.04779): Conservative Q-learning to reduce distribution shift error in offline value estimation.

### On-Policy RL

* [SPPO](https://arxiv.org/pdf/2604.08865v1): Sequence-level PPO with a learned value function to enable single-sample-per-problem learning.  
* [SPO](https://arxiv.org/pdf/2509.13232): Similar to SPPO but with advantage normalization over the batch.  
* [AGRO](https://arxiv.org/pdf/2503.19612v1): Unifies DRO-like off-policy learning with on-policy learning.  
* [Leverage On-Policy](https://arxiv.org/pdf/2404.14367): Discussion of the use-cases and trade-offs between on/off-policy training, positive/negative likelihood gradients, and mode-seeking losses.

### Meta-continual Learning (MCL):

* [MCL](https://arxiv.org/pdf/1806.06928): Shows that forgetting can be minimized by replaying (not included in previous inner-loop updates) samples from previous tasks during meta-learning.  
* [VC-BML](https://proceedings.neurips.cc/paper/2021/hash/cdd0500dc0ef6682fa6ec6d2e6b577c4-Abstract.html) / [SB-MCL](https://arxiv.org/pdf/2405.18758): Bayesian dynamics and updates in meta-continual learning to prevent forgetting.  
* [SRWM](https://proceedings.mlr.press/v162/irie22b.html): Self-Referential Weight Matrix. A full network parameterized by a single matrix that updates its state using the delta-rule and non-linear QKV.  
* [ACL](https://arxiv.org/pdf/2312.00276): A very general instantiation of meta-continual learning (a forward-model directly outputs weight changes). Uses an SRWM architecture (no convergent extrapolation). Emphasizes replay of past tasks in meta-objective to learn not to forget.  
* [MCL Attention](https://arxiv.org/pdf/2408.03219): Meta-continual learning by calculating weight updates using a meta-trained attention/transformer over (encoded) previous examples.  
* [MC: as SM](https://arxiv.org/pdf/2310.11952): 

### Free Energy

* [PC-MRL](https://arxiv.org/abs/2510.22039v1): Shows that free energy minimization in meta-RL can create better feature representations and outperform standard meta-RL.  
* [Task Inference](https://arxiv.org/pdf/1905.06424): An example of Bayesian RL where the task belief is updated with formalized methods similar to free energy minimization.  
* [DVRL](https://proceedings.mlr.press/v80/igl18a/igl18a.pdf): A variational approach to updating task beliefs using a generative model.  
* [Efficient MRL](https://arxiv.org/pdf/1903.08254): Efficient meta-RL by creating a latent representation of the task instead of relying only on policy gradients.  
* [Predictive RL](https://proceedings.iclr.cc/paper_files/paper/2024/file/0b5eb45a22ff33956c043dd271f244ea-Paper-Conference.pdf): Auxiliary free-energy-like prediction objectives improve RL performance and yield neural networks with resemblance to hippocampus activity in the brain.

### Eligibility Traces

* [GPOMDP](https://arxiv.org/pdf/1106.0665): Online infinite-horizon reinforcement learning using REINFORCE updates over exponentially decaying eligibility traces of gradients.   
* [RL Eligibility](https://chizkidd.github.io/RL-Sutton-Barto-notes/chapters/ch12-eligibility-traces.html): The backward view of the TD(lambda) RL algorithm using eligibility traces for online learning.  
* [Neuro ET](https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/): Connections between eligibility traces in 3-factor learning from neuroscience and reinforcement learning.

## Technical Details

Our methods are largely based on modernizing/improving/combining/scaling existing methods, rather than blazing a completely new path. This reduces unknowns and makes the research less risky. While it may seem like less of a moat than a single radical breakthrough, the sum of all of the insights, combinations, and improvements can be just as novel and impactful. In fact, many of machine learning’s historic breakthroughs could be described in this form: Original ChatGPT is just GPT-1 with SFT/RLHF and scaling; AlphaGo is just [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon) with deeper search and scaling; AlexNet is just a ConvNet with ReLU and scaling.

Concretely, our method is to meta-train a language model to perform gradient-based continual learning. Through novel combinations and improvements of sequence modelling, offline/online RL, meta-learning, scale, and curricula, we will create a modern LLM with the ability to learn and improve during deployment. 

Our research agenda consists of 4 steps. Note that compute cost estimates below are based on a 1B parameter model (and should scale about linearly to larger models).

### 1\. Sequence-Modelling Pretraining 

This is the scalable pretraining phase of building continual-learning foundation models. When an existing model is used as the starting point for training, this phase may not be completely necessary (or some weights might be frozen). 

During this phrase, we treat chunked autoregressive sequence modelling (next-token prediction) as the meta-learning objective, similar to TTT-E2E. The inner loop consists of iterating over chunks of readily-available long-context sequences and updating fast weights after each. Unlike traditional meta-learning with a clear meta-train/meta-test split, this version is autoregressive: each chunk acts as both an inner-loop meta-train example relative to future chunks and outer-loop meta-test example relative to previous chunks. The idea is that this will prime the model for later meta-continual learning.

Our breakthrough in this stage is the replacement of the end-to-end gradients used by TTT-E2E with first-order gradients and hypergradients. This makes training significantly faster and easier because it does not require gradients-of-gradients, and is analogous to going from the practically infeasible MAML meta-learning algorithm to the standard-practice FOMAML. With a few other small improvements (such as a novel inner optimizer and LoRA fast weights), we get our method [iTTT](https://aklein.bearblog.dev/ittt/). This change also turns O(N) training memory into O(1), while maintaining O(1) inference memory and O(N) inference compute complexity.

**Data:** Standard long-context sequence-modelling datasets (synthetic, [FinePDFs-edu](http://HuggingFaceFW/finepdfs-edu), [Longmino](https://huggingface.co/datasets/allenai/dolma3_longmino_pool), [Institutional Books](https://huggingface.co/datasets/institutional/institutional-books-1.0))

**Baselines:** Full attention, TTT-E2E

**Benchmarks:** Perplexity, Retrieval, many-shot ICL

**Cost:** 2 person weeks of labor, 100 H100 days (100B tokens) of compute

**Failure Modes:**

* Relatively low performance in the many-pretraining-tokens regime  
  * Not a big deal since this is mostly for transfer learning

**Deliverable:** A sequence modelling architecture in a new scaling regime (O(1) training memory) with convergent context length extrapolation and priming for later meta-continual learning.

**Value Level:** Publishable and might see interest as an open-source model, but not much commercial value.

### 2\. Off-policy MCL

This stage is where the real meta-continual learning starts. By meta-learning on multi-task multi-episode trajectories, the model learns to make use of past experience to improve later performance. In the realm of LLM pretraining recipes, this is comparable to off-policy post-training (SFT, DPO).

Our key insight at this stage is that we can transfer the learning from the pretraining stage to new inner and outer reward functions. While TTT-E2E and iTTT specifically use next-token prediction (NTP) as their loss, we can also meta-learn using SFT losses (for ICL-like supervised learning), off-policy RL losses, preference losses, and reward-modelling losses. These losses could all be meta-learned by the same model (though some parameters like learning rates may not transfer well between loss functions). Just like how iTTT takes a gradient step after every chunk, we will take a gradient step after every episode.

Our offline RL loss will be DRO-V, since unlike AWR it actually allows for negative weights on the logit gradients. DRO-V also appears to fulfill the universality conditions noted by MACAW. SFT losses can use the standard next-token-prediction cross-entropy, preference can be DPO-like, and RM losses can use standard squared loss.

Training will be performed over synthetic multi-task multi-episode trajectories. These data trajectories can be readily constructed at giga-token scale using existing publicly-available data, making training data scalable. We may also mix in pretraining data (treating chunks as SFT episodes) to maintain general knowledge in the model. The success of methods like MetaICL and TTT-NN make us confident that trajectories composed of synthetically clustered tasks will actually improve late-trajectory performance. Training also inherits the complexity and cost of iTTT, making it compute scalable.

While this seems like an obvious thing to do (in many ways this is just LLM-scale MACAW), it seems to be the case that nobody else has thought to *just do the thing* of collecting massive amounts of open-source offline RL (or even just SFT) data and meta-training an LLM on them. 

Forgetting can be improved by interleaving tasks and meta-training (but not inner-loop updating) on unseen examples from previous tasks and general knowledge problems.  Since a frozen teacher model is needed for DRO-V anyway, we can also make simple architectural changes to incorporate it into inference and improve forgetting (something like [ControlNet](https://arxiv.org/abs/2302.05543)).

At this point, we may also start performing free energy minimization using a generative modelling loss (like next-token-prediction cross-entropy) over inputs, observations, and feedback with the hope of increasing sample efficiency further. 

It should be noted that while our meta-learning trajectories will only be of a certain length, our inner-loop update rule is actually still gradient descent over a loss function. This means that even after the meta-learning horizon is exceeded, our model’s performance will continue to converge towards better performance (as has been shown with MAML). This is what makes this method [actually useful compared to alternatives with no guarantees](https://www.lesswrong.com/posts/9rCTjbJpZB4KzqhiQ/you-can-t-imitation-learn-how-to-continual-learn).

**Data:** See data section below

**Baselines:** In-context learning, non-meta-learned PEFT

**Benchmarks:** few/many-shot supervised learning, Multi-episode RL task regret, forgetting metrics, TTT-Discover style optimization. See CASCADE, CaRE, MemRL, IDAQ, LifelongAgentBench

**Cost:** 4 person weeks of labor, 50 H100 days (50B tokens)

**Failure Modes:**

* Low inter-episode mutual information after just clustering (previous reward samples must be informative about what future actions should be)  
  * *Backup:* See data section  
* DRO-V is unstable or has poor sample efficiency  
  * *Backup:* MACAW-like AWR loss function  
* Free energy minimization does not transfer to downstream improvement  
  * *Backup:* Use iTTT version that computes some high-order components (already implemented/tested), or use full E2E gradients for a portion of training.  
* Performance is weaker than attention-based ICL  
  * *Backup:* Focus on long trajectories where attention becomes infeasible.

**Deliverable:** An LLM that behaves like a super in-context supervised/RL learner with constant-state scaling and semi-guaranteed extrapolation

**Value Level:** Commercially useful for short-episode tasks. 

### 3\. On-policy MCRL

Here, we move from off-policy multi-episode meta-continual RL to on-policy. 

We will probably use an SPPO-like policy gradient loss. To ease the transition, we could start by fine-tuning a V head on a few samples while freezing the rest of the model. The biggest question is whether switching to an on-policy RL loss will be too big of a shift away from the model’s previous DRO adaptation. If policy gradient losses are too much of a shift, we can use the more DRO-like online AGRO algorithm (with a learned baseline).

We can continue to mix in other loss types (which don’t have offline/online variants) and free energy minimization. We can also incorporate hidden COT reasoning during this stage. Previous architectural things like iTTT updates and teacher-mixing architectures can stay the same.

Unless inter-episode reward interactions are considered, this phase will entice the model to greedily maximize the reward of each episode, ignoring the benefits of exploration. A simple improvement would be to use regret-based rejection-sampling at the trajectory level. However, exploration is important in practice and worth a closer look. A complicating improvement would be to use two extra forward/backward passes (one for the gradients of the discounted exploration-aware loss in the slow weights, and then one for the greedy loss on the fast weights to get ready for the next chunk).

**Data:** Online RL environments, probably agentic. There are many startups building these, and we could also create simple environments (see data section below)

**Baselines:**  In-context learning, non-meta-learned PEFT

**Benchmarks:** Same as stage 2\.

**Cost:** ?

**Failure Modes:**

* The model breaks when going from DRO to policy gradient  
  * *Backup:* Keep using a DRO-like loss (online ARGO)  
* Not enough unique training environments to generalize  
  * If nothing else we should be able to scale on RLVR  
* Simple ejection sampling is not enough to encourage exploration  
  * Incorporate more complicated trajectory-level losses

**Deliverable:** A stronger version of the model from stage 2, with less distribution shift.

**Value Level:** A more commercially viable model for short-episode tasks.

### 4\. Online Infinite-Horizon RL (Moonshot)

At this stage, we ditch the strict multi-episode framing from previous stages and move into the infinite-horizon regime. This means that the model never reaches the ‘end’ of an episode during deployment, and must update online.

For this, we use GPOMDP-like discounted eligibility tracing while keeping details like the inner-loop optimizer and inner-loop learning rates from previous stages the same. This provides an online estimate of inner-loop REINFORCE updates, but does not strictly account for second-order gradient interactions. This is very similar to the previous stage. In fact, the previous stage is a special case of this stage where the discount factor is zero (when episodes are bandit-style). This should make warm-starting after previous training extremely effective. We can also incorporate modern policy gradient methods to improve over vanilla GPOMDP. 

We can also continue doing things like free energy minimization and reward model training.

**Data:** Extremely long (probably agentic) on-policy trajectories with intermediate rewards and multiple interactions. This would have been impossible before the recent emergence of long-horizon systems like Claude Code, and will hopefully become possible for other types of text-based knowledge work soon. There is also a fledgling industry of companies collecting exactly these types of trajectories and feedback.

**Baselines:** Text-based agentic harnesses, other continual learning companies

**Benchmarks:** LifelongAgentBench, ?

**Cost:** A lot

**Failure Modes:**

* The meta-learning outer-loop is sample-inefficient, making cost the prohibitive  
* Model is too weak in order to collect meaningful rollouts during the start of training  
* Environments and rewards are too difficult to collect  
* GPOMDP bias and/or variance is too strong and training is unstable  
* Credit assignment is too noisy or indirect

**Deliverable:** A model that can learn from experience during infinite-horizon deployments without clear episode boundaries. Very close to what pundits call ‘true continual learning’ in the modern sense.

**Value Level:** Commercially useful for a variety of applications as a model that truly “learns on the job”.

## Data

### SFT / Supervised Datasets

This can be basically any textual SFT or chat data, which is extremely abundant.

### Offline RL Datasets

Huggingface already contains billions of tokens worth of labelled text datasets for offline reinforcement learning.

**Generic:**

* [https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data](https://huggingface.co/datasets/RLHFlow/Mistral-PRM-Data)  
* [https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data](https://huggingface.co/datasets/RLHFlow/Deepseek-PRM-Data)  
* [https://huggingface.co/datasets/llm-blender/Unified-Feedback](https://huggingface.co/datasets/llm-blender/Unified-Feedback)  
* [https://huggingface.co/datasets/OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/lvwerra/stack-exchange-paired](https://huggingface.co/datasets/lvwerra/stack-exchange-paired)  
* [https://huggingface.co/datasets/stanfordnlp/SHP-2](https://huggingface.co/datasets/stanfordnlp/SHP-2)  
* [https://huggingface.co/datasets/Mai0313/prm800k](https://huggingface.co/datasets/Mai0313/prm800k?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/TIGER-Lab/AceCodePair-300K](https://huggingface.co/datasets/TIGER-Lab/AceCodePair-300K)  
* [https://huggingface.co/datasets/openai/summarize\_from\_feedback](https://huggingface.co/datasets/openai/summarize_from_feedback?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/ibm-research/ToolRM-train-data](https://huggingface.co/datasets/ibm-research/ToolRM-train-data)  
* [https://huggingface.co/datasets/RioLee/ToolPref-Pairwise-30K](https://huggingface.co/datasets/RioLee/ToolPref-Pairwise-30K)  
* [https://huggingface.co/datasets/nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback\_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)  
* [https://huggingface.co/datasets/peiyi9979/Math-Shepherd](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)  
* [https://huggingface.co/datasets/abacusai/MetaMath\_DPO\_FewShot](https://huggingface.co/datasets/abacusai/MetaMath_DPO_FewShot)  
* [https://huggingface.co/datasets/interstellarninja/tool-calls-dpo](https://huggingface.co/datasets/interstellarninja/tool-calls-dpo)  
* [https://huggingface.co/datasets/Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)  
* [https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)  
* [https://huggingface.co/datasets/prometheus-eval/Feedback-Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection)

**SWE:**

* [https://huggingface.co/datasets/hao-li/AIDev](https://huggingface.co/datasets/hao-li/AIDev)  
* [https://huggingface.co/datasets/bigcode/the-stack-github-issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues?utm_source=chatgpt.com)  
* [https://huggingface.co/datasets/nebius/SWE-agent-trajectories](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)  
* [https://huggingface.co/datasets/declare-lab/PathFinder-600K](https://huggingface.co/datasets/declare-lab/PathFinder-600K)  
* [https://huggingface.co/datasets/kevinpro/R-PRM](https://huggingface.co/datasets/kevinpro/R-PRM)  
* [https://huggingface.co/datasets/SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat)  
* [https://huggingface.co/datasets/ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview?utm_source=chatgpt.com)  
* [https://zenodo.org/records/6900648](https://zenodo.org/records/6900648?utm_source=chatgpt.com)

**Feedback:**

* [https://huggingface.co/datasets/nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)  
* [https://huggingface.co/datasets/openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback)   
* [https://huggingface.co/datasets/prometheus-eval/Feedback-Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) 

## Combining Episodes

Since nearly all available data takes the form of independent episodes, we need to figure out how to concatenate them into multi-episode trajectories. The key constraint is that early episodes need to be informative about later episodes.

A few ideas:

* **Semantic Nearest-Neighbors.** Simple. Used by TTT-NN.

* **SIFT:** Outperforms NN (when you don’t know the answer), used by TTT-SIFT and TTC-RL.

* **Loss Minimization.** Similar in concept to [EntropyLong](https://arxiv.org/abs/2510.02330) for creating long-context corpora, except we search for examples that minimize the loss of later episodes (RL, cross-entropy, whatever we are minimizing).

* **Gradient Attribution.** Something like [TrackStar](https://arxiv.org/abs/2410.17413v2) to find examples that have aligned gradients with respect to the loss (in our case the gradients of the fast weights).

Another unanswered question is how to *order* the episodes within a trajectory.

To reduce off-policy distribution shift, we can use importance sampling. This means that when an input has multiple reward-labelled outputs (such as in preference datasets), we choose or sample the output that the model is most likely to produce. Ideally, this would also give us a realistic curve of lower rewards early in a trajectory and higher rewards later.

Mixing independent or less related episodes, as well as experience replay (with held-out episodes from previously seen clusters) can massively [help with catastrophic forgetting](https://arxiv.org/pdf/1806.06928). To minimize inference-time distribution shift, gradients from the anti-forgetting episodes could be masked from the rest (similar to classic meta train/test splits).

## RL Data Sources

To continue scaling, we will need to collect more multi-episode examples of text-based reinforcement learning with labelled rewards.

An important thing to note is that, at least during early training, the reward function doesn’t necessarily need to be *correct* (as in the rewards that humans would give), it just needs to be *consistent* (so that the model has a useful signal to learn from). This implies that data augmentation strategies like sign-flipping rewards could be useful. 

Some potential sources:

* Synthetic data from verifiable domains (math, code)  
* Personalized preference and recommendation datasets (especially good because reward functions are personalized and specific)  
* Multi-turn interaction/chat data (extract rewards using sentiment classification on next message)  
* LLM-as-a-judge (same model for entire trajectory for consistency, many different models on different trajectories for different reward functions). Can improve reward sample mutual information by iterating through each trajectory and placing previously sampled episode-reward pairs in context, or just by adding a randomly generated system prompt.  
* Linear combinations of different reward models [can produce proxies of user-specific rewards](https://huggingface.co/datasets/google/rfm-rm-as-user-dataset)  
* [self rewards](https://arxiv.org/abs/2401.10020) or [test-time scaling distillation](https://arxiv.org/abs/2504.16084)  
* [Delta learning](https://arxiv.org/abs/2507.06187)  
* Accepted/rejected pull requests and edits for coding  
* Board games  
* Text/terminal games  
* RL gyms with small observation spaces  
* Simple procedural RL environments  
* MCP servers and [synthetic tasks](https://arxiv.org/abs/2605.18703)  
* Games generated/vibe-coded by coding agents

# Technical Theses (rambling)

# Other Theses (rambling)

Not all of these claims are unique or relevant to continual learning (some are), but set the precedent for how we think and will inform future research.

1. Optimization loops, especially when nested, are the most powerful force in the universe. As evidence, look at nature: for 4 billion years there was a single optimization loop (evolution) until humans came along with a second one (learning within a single lifetime) which put things on a qualitatively different trajectory. Over long time periods, evolution also appears to assert poorly understood higher-order “evolving to evolve” effects, which could be a hidden extra optimization level and critical to the progress of evolution.

2. Monkeys with typewriters are all you need. Inside an optimization loop with sufficient time  and an oracle fitness function, every distribution with full coverage, from an LLM to a uniformly random character generator, will eventually discover everything from the code for AGI to the cure for cancer. Everything we’re doing now merely speeds up the search: better LLMs are better sampling distributions, better test-time scaling strategies are better metaheuristic transition functions, etc.

3. Interaction feedback should be richer than sparse scalar rewards, [to maximize the amount of information gained](https://youtu.be/Ount2Y4qxQo?si=XCN68c3rV8Lq08hg&t=1165).

4. On-policy learning is critical for real-world performance, since it enables error-correction, incentivizes introspection (such as [knowing when you don’t know](https://arxiv.org/abs/2509.25760v1)), and reduces the training-inference distribution shift from teacher forcing.

5. Mode-seeking losses (RL, on-policy distillation, GAN) can create much more parameter-efficient models than mode-covering losses (cross-entropy, ELBO).

6. When test-time scaling is an option, the reward/verifier is more important than the generator (see point 2). [Nested verifiers](https://arxiv.org/pdf/2511.22570v1) (which verify the rationale of lower-level reward models) could be a good solution to better verifiers.

7. Architectures with [serial circuit complexity](https://arxiv.org/abs/2507.12549) will supersede parallel architectures (transformers, SSMs) in the near(?) future, at least for small models. This may coincide with new computing hardware/infrastructure that focuses on higher serial throughput at the cost of parallelism.

8. [Test-time RL](https://arxiv.org/abs/2601.16175) is the test-time scaling method of the future. It can also be meta-learned (on both the generator and verifier level). 

9. Compression (both referring to encoder models and sequence-modelling/continual-learning) should allocate state capacity towards mutual information with downstream tasks, rather than uniform entropy reduction over the input.

10. In sequence modelling and continual learning, as long as the state capacity is large enough to encode the entire past context, then the exact mechanisms of encoding/retrieving it don’t really matter if you have a sufficiently large model. All that matters is making sure that all of the information describing the past is still theoretically accessible in the state.

