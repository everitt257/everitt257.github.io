---
layout: post
title: Learning Theory
---


$\newcommand{\E}{\operatorname{\mathbb{E}}}$
$\newcommand{\P}{\operatorname{\mathbb{P}}}$
$\newcommand{\R}{\operatorname{\mathbb{R}}}$

## Learning theory
### Generalization error
**Questions** that we might want to ask:
- Most learning algorithms fit their models to the training set, why should doing well on the training set tell us anything about generalization error?
- Can we relate error on the training set to generalization error?
- Are there conditions under which we can actually prove that learning algorithms will work well?

**Hoeffding inequality** (Perhaps the most important inequality in learning theory)
$$
\P(\frac{1}{n}\sum_{i=1}^{n}(Z_i-\E(Z_i)) \geq t) \leq exp(-\frac{2nt^2}{(b-a)^2})
$$

1. Markov's inequality:
$$\operatorname{\mathbb{P}}(Z\geq t) \leq \frac{\operatorname{\mathbb{E}}[Z]}{t}$$
    * Let Z be an non-negative r.v and t > 0
    * Link to the [proof](http://cs229.stanford.edu/extra-notes/hoeffding.pdf).

2. Chebyshev's inequality: A consequence of Markov's inequality
$$
\begin{split}
\operatorname{\mathbb{P}}(Z \geq \operatorname{\mathbb{E}}+t\; or \;Z \leq \operatorname{\mathbb{E}(Z)}-t) &= \operatorname{\mathbb{P}}((Z - \operatorname{\mathbb{E}(Z)})^2 \geq t^2) \\
& \leq \frac{\operatorname{\mathbb{E}}[(Z-\operatorname{\mathbb{E}}(Z))^2]}{t^2} = \frac{Var(Z)}{t^2}
\end{split}
$$
    * What this means is that average of random variables with finite variance converges to their mean. Given enough samples.
3. Chernoff bounds
    * Essentially it's saying, for any $\lambda$, we have $Z \geq \operatorname{\mathbb{E}}[Z] + t$ if and only if $e^{\lambda Z} \geq e^{\lambda \operatorname{\mathbb{E}}[Z] + \lambda t}$ or $e^{\lambda (Z-\operatorname{\mathbb{E}}[Z]) \geq e^{\lambda t}}$
    * Chernoff bounds use moment generating functions in a way to give exponential deviation bounds.
    * The Chernoff bounds proof is done based on the combination of the moment function of random variable and the markov inequality. 
    * It also works well with r.v that is the sum of i.i.d random variables. For example it gives bounds in exponential form for r.v that is the sum of i.i.d random variables. 
4. Jensen's inequality
    * The jensen's inequality applies to convex functions. In our case, the most common ones are $f(x) = x^2$, $f(x) = e^x$ and $f(x) = e^{-x}$
$$f(\E(x)) \leq \E (f(x))$$
in terms of probability.
$$f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)$$
also in terms of convex function, for $t \in [0,1]$.
    * Again, Jensen's inequality for probability is just the infinite sums of integral of normal Jensen's inequality. 
5. Symmetrization
    * A technique in probability. This along with Jensen's inequality is used to prove the Hoeffding's lemma.
6. Hoeffding's lemma
$$\E[exp(\lambda(Z-\E[Z]))] \leq exp(\frac{\lambda^2(b-a)}{8})$$
    * Let $Z$ be a bounded random variable with $Z \in [a,b]$, then the above holds for all $\lambda \in \R$

7. Hoeffding's inequality conclusions:
    - Point 2 is proven via point 1.
    - Point 3 is proven via point 1 in the exponential case. And its generalization applies well to sums of r.vs.
    - Point 6 is proven via point 4 and point 5. It serves as intermediate part for proving the Hoeffding's inequality.
    - To prove Hoeffding's inequality, we mainly use **Hoeffding's lemma** and **Chernoff bounds** to do so. Namely we use point 3 and 6 to do so.
    - Essentially one can interpret Hoeffding's inequality like this: 
        * The probability of of our estimate for the mean far from the the true mean is small as long as the sample size is large enough. Given each sample is i.i.d. 

**Empirical risk/error**

For a binary classification problem, define empirical risk as:
$$\hat{\xi}(h) = \frac{1}{m}\sum_{i=1}^{m}1\{h(x^{(i)}) \neq y^{(i)}\}$$

**Generalization error**

- Define generalization error under the **PAC** assumptions.
    - traning and testing on the same distribution $D$
    - independently drawn training examples
- Define generalization error to be the probability that our model $h$ will misclassify our traning sample $(x,y)$

$$\xi(h) = P_{(x,y)\sim D}(h(x) \neq y)$$

- Note the difference between $\xi(h)$ and $\hat{\xi}(h)$

- Empirical risk minimization(ERM)
$$\hat{\theta} = arg\; \min_{\theta}\hat{\xi}(h_{\theta})$$
    - The most "basic" learning algorithm
    - Define $H$ as the set of all hypothesis used in certain class of algorithm
    - The process of empirical risk minimization is then a way of picking the best hypothesis $h$ in the set $H$
$$\hat{h} = arg\; \min_{h\in H}\hat{\xi}(h)$$

**Example**
**Conclusions**