# Preliminary meta-learning scaling law

## General form

Let $M$ represent meta-training compute, and $n$ represent the number of examples seen for a task.

The two-dimensional scaling-law model is

$$
\boxed{
L(M,n)=L_\infty+
\left[L_0(M)-L_\infty\right]
\left[1+\left(\frac{g(M)n}{n_0}\right)^\gamma\right]^{-\delta}
}
$$

with

$$
L_0(M)=L_0(\infty)+A_0
M^{-\alpha_0},
\qquad
g(M)=M^\beta
$$

Here, $L_0(M)$ represents the zero-shot loss and follows a standard LLM pretraining scaling law. $g(M)$ is the learning efficiency, with higher $g$ indicating faster task learning.

$L_\infty$ represents the minimum loss that can be attained as $M,n\to\infty$.

The adaptation function is

$$
F(x)=\left[1+\left(\frac{x}{n_0}\right)^\gamma\right]^{-\delta}
$$

which monotonically decreases from 1 to 0 as $x>=0$ increases. It models how the loss decreases as task examples are seen. This generalized function contains two simpler functions as special cases:

- $\gamma=1$ gives the shifted power function $F(x)=(1+x/n_0)^{-\delta}$.
- $\delta=1$ gives the log-sigmoid function $F(x)=x^\gamma/(x^\gamma+n_0^\gamma)$.

## Degrees of freedom

| Variable | Interpretation | Constraint |
|---|---|---|
| $L_0(\infty)$ | Asymptotic zero-shot loss as $M\to\infty$; the usual irreducible-loss term in scaling laws. | $L_0(\infty)\geq 0$ |
| $A_0$ | Excess zero-shot loss at the start of training: $A_0\approx L_0(M_0)-L_0(\infty)$. | $A_0>0$ |
| $\alpha_0$ | Scaling exponent governing how zero-shot loss approaches $L_0(\infty)$. | $\alpha_0>0$ |
| $L_\infty$ | The minimum loss approached as $n\to\infty$ for any $M$. | $0\leq L_\infty\leq L_0(\infty)$ |
| $\beta$ | Meta-learning scaling exponent. $\beta>0$ means later checkpoints adapt more efficiently; $\beta<0$ less efficiently.
| $n_0$ | Characteristic example scale; sets the horizontal position of the adaptation curve. | $n_0>0$ |
| $\gamma$ | Response-shape exponent controlling the early-to-middle transition and bend sharpness on a log-$n$ axis. | $\gamma>0$ |
| $\delta$ | Tail-shape exponent. At large $n$, $L-L_\infty\propto[g(M)n]^{-\gamma\delta}$. | $\delta>0$ |

The example count that closes half of the available loss gap is

$$
n_{\text{half}}(M)=\frac{n_0}{g(M)}
\left(2^{1/\delta}-1\right)^{1/\gamma}.
$$

This derived quantity may be easier to interpret than $n_0$ by itself.

## Data and evaluation protocol

The model was fitted independently to:

- **Meta-learned:** `aklein4--horizon-v2_piano-scaled`
- **LoRA:** `fresh/oloop-lora-llama3p2-1b-pre`

The fit used all 17 numeric checkpoints from $M=50$ through $1600$, but only the eight example levels

$$
n\in\{0,1,2,4,8,16,32,64\}.
$$

This gives 136 fitted observations per run. The reported loss at each point is the average across the 12 evaluation benchmarks. Fits use unweighted nonlinear least squares.

Evaluation uses three held-out settings:

1. **Example-count interpolation:** fit $n=\{0,1,4,16,64\}$ and evaluate $n=\{2,8,32\}$.
2. **Checkpoint interpolation:** fit alternating checkpoints and evaluate the intervening checkpoints.
3. **Example-count extrapolation:** fit all $n\leq64$ and evaluate $n=\{128,256,512,1024\}$.

Pooled RMSE is the root mean square over the meta-learned and LoRA residuals, not the arithmetic mean of their two RMSEs.

## Fitted laws

### Meta-learned

The fitted components are

$$
L_0(M)=1.714176+0.083466
\left(\frac{M}{50}\right)^{-0.922824},
$$

$$
g(M)=\left(\frac{M}{50}\right)^{0.060063},
$$

and

$$
L(M,n)\simeq L_0(M)
\left[1+\left(
\frac{g(M)n}{1.644753}
\right)^{0.669100}\right]^{-0.257402}.
$$

The fitted $L_\infty$ is numerically indistinguishable from zero. From checkpoint 50 to 1600, $g(M)$ increases by a factor of 1.231, implying approximately a **23% increase in effective example use**. The fitted half-gap point at checkpoint 50 is $n_{\text{half}}(50)\approx82.9$ and at checkpoint 1600 is $n_{\text{half}}(1600)\approx67.3$.

### LoRA

The fitted components are

$$
L_0(M)\simeq1.771792
\left(\frac{M}{50}\right)^{-0.000990},
$$

$$
g(M)=\left(\frac{M}{50}\right)^{-0.009348},
$$

and

$$
L(M,n)\simeq L_0(M)
\left[1+\left(
\frac{g(M)n}{1.865445}
\right)^{1.170616}\right]^{-0.137673}.
$$

Again, $L_\infty$ goes to zero. The tiny $\alpha_0$ makes $L_0(M)$ nearly constant in practice. Likewise, $g(M)$ decreases by only about 3.2% from checkpoint 50 to 1600, so LoRA is consistent with little or no scaling of its adaptation rate. Its fitted half-gap point at checkpoint 50 is $n_{\text{half}}(50)\approx136.8$.

## Results

### Fit quality

| Evaluation | Meta-learned RMSE | LoRA RMSE | Pooled RMSE |
|---|---:|---:|---:|
| Training, $n\leq64$ | 0.006982 | 0.006802 | 0.006892 |
| Example-count interpolation | 0.006599 | 0.009688 | 0.008289 |
| Checkpoint interpolation | 0.007802 | 0.006936 | 0.007381 |
| Extrapolation to $n>64$ | 0.012707 | 0.014503 | 0.013635 |

## Ablations

### Simpler adaptation functions

Constraining the generalized adaptation function to either nested seven-variable family worsens every pooled metric relative to the generalized form.

| Response | Vars/run | Training | $n$-interp. | $M$-interp. | $n>64$ extrap. |
|---|---:|---:|---:|---:|---:|
| Power, $\gamma=1$ | **7** | 0.009301 | 0.010482 | 0.009742 | 0.044717 |
| Log-sigmoid, $\delta=1$ | **7** | 0.008545 | 0.011193 | 0.008898 | 0.105992 |
| **Generalized** | 8 | **0.006892** | **0.008289** | **0.007381** | **0.013635** |

The generalized form reduces pooled RMSE by approximately 17--21% on fitting and interpolation and by about 70% on $n>64$ extrapolation, relative to the best simpler family for each metric.

### Removing $n_0$

Fixing $n_0=1$ while retaining $g(M_0)=1$ forces the response transition to occur at one example. This substantially degrades the two simpler families:

| Response | $n_0$ | Vars/run | Training | $n$-interp. | $M$-interp. | $n>64$ extrap. |
|---|---:|---:|---:|---:|---:|---:|
| Power | fitted | 7 | 0.009301 | 0.010482 | 0.009742 | 0.044717 |
| Power | fixed to 1 | 6 | 0.017132 | 0.015772 | 0.016928 | 0.036823 |
| Log-sigmoid | fitted | 7 | 0.008545 | 0.011193 | 0.008898 | 0.105992 |
| Log-sigmoid | fixed to 1 | 6 | 0.069698 | 0.071035 | 0.064411 | 0.286155 |

The lower tail RMSE of the scale-free power ablation comes with much worse fit and interpolation errors.

If a free amplitude is added to $g(M)$, that amplitude is exactly confounded with $n_0$, so it merely restores the removed variable under a different name.

### Adding a checkpoint offset to $L_0(M)$

The alternative

$$
L_0(M)=L_0(\infty)+A_0
\left(M+M_{\rm off}\right)^{-\alpha_0}
$$

was tested extensively in the nested power and log-sigmoid response sweeps. It slightly improved example-count interpolation but worsened checkpoint interpolation:

| Response | $M_{\rm off}$ | $n$-interp. | $M$-interp. | $n>64$ extrap. |
|---|---:|---:|---:|---:|
| Power | absent | 0.010482 | **0.009742** | 0.044717 |
| Power | fitted | **0.010382** | 0.010448 | **0.044166** |
| Log-sigmoid | absent | 0.011193 | **0.008898** | **0.105992** |
| Log-sigmoid | fitted | **0.011135** | 0.009644 | 0.106004 |

Meta-learning drives $M_{\rm off}$ to zero. For LoRA, $M_{\rm off}$ is weakly identified and can run to very large values while compensating with $\alpha_0$, effectively making $L_0(M)$ constant. The offset therefore adds instability without a consistent validation improvement and is omitted from the primary law.

### More flexible $g(M)$

Adding an asymptotic floor and then a checkpoint offset to $g(M)$ was also tested:

$$
g(M)=g_\infty+(1-g_\infty)
\left(\frac{M+M_{g,{\rm off}}}{M_0+M_{g,{\rm off}}}\right)^{-\alpha_g}.
$$

In the nested response sweeps, these additions changed pooled interpolation RMSE by only a few $10^{-4}$ while adding one or two variables. Their parameters frequently became weakly identified. A constant $g(M)$ was already competitive for LoRA, while the single exponent $\beta$ captures the clearer meta-learning trend. The one-parameter power form is therefore the preferred common specification.
