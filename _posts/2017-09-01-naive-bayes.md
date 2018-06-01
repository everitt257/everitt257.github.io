---
layout: post
title: Naive Bayes
category: blog
tags: algorithm
---

# ML 2 Generative Learning
In this notebook we make distinction between discriminative learning and generative learning algorithms.
## Conditional Probability
The conditional probability is fundemental in statistc inference.

$$p(x \mid y) = \frac{p(x,y)}{p(y)}$$

Since $p(x,y) = p(y,x) = p(y \mid x)p(x)$ by the bayes rule, we can also write the above rule as

$$p(x \mid y) = \frac{p(y \mid x)p(x)}{p(y)}$$

Which is the Bayes rule below. The term $p(x)$ is refered as the *class prior* while $p(x \mid y)$ is refered as the *posterior*.
## Tricks with conditional probability and Bayes
- Suppose we want to measure the pmf/pdf of multiple random variables, say $p(x,y,z)$
    - Sometimes it can be difficult to measure $p(x,y,z)$ directly, **Bayes to the rescue**
        - $p(x,y,z) = p(x,y \mid z)p(z) = p(x,z \mid y)p(y) = p(y,z \mid x)p(x)$ any combination you like~
        - you can break it down further like this $p(x,y,z) = p(x,y \mid z)p(z) = p(x \mid y,z)p(y \mid z)p(z)$
        - and if $x,y,z$ are independent, you would have $p(x,y,z) = p(x)p(y)p(z)$, looks sweet doesn't it
    - Sometimes we concern about joint conditional probability, say $p(x,y \mid z)$
        - as before we may rewrite this conditional probability in terms of other combination
        - so $p(x,y \mid z) = \frac{p(x,y,z)}{p(z)} = \frac{p(y,z \mid x)}{p(z)}$, which may be easier to compute?

## Bayes Rule
The bayes rule is used to derive the posterior distribution on y given x

$$p(y\mid x) = \frac{p(x\mid y)p(y)}{p(x)}$$

Note we don't actually need to compute p(x) in order to make prediction. The prediction can be made sole based on:

$$
\begin{split}
\arg\max_{y}p(y\mid x) &= \arg\max_{y}\frac{p(x\mid y)p(y)}{p(x)} \\
&= \arg\max_{y}p(x\mid y)p(y)
\end{split}
$$

Given $p(y)$ the **class priors**. The features $x$ can be broken down into $x1, x2, x3 ... x_n$. This means to calculate $p(x\mid y)$ we can rewrite as $p(x_1,x_2,x_3...x_n\mid y)$. If we assume the features are indepently distributed, we can rewrite $p(x \mid y)$ as

$$p(x_1,x_2,x_3...x_n\mid y) = p(x_1\mid y)p(x_2\mid y)p(x_3\mid y)...p(x_n\mid y)$$

All we need to do is selecting very good features to predict. Also I like to point out these indivisual distributions can be guessed or learned. Another thing is that even if these feautures are not indepent, the equation still holds for most of the time.

## Gaussian discriminant analysis
Before the derivation, we need to understand that what the term **probability of data** is.
>The probability of the data is given by
$p(\vec{y}\mid X; \theta)$. This quantity is typically viewed a function of $\vec{y}$ (and perhaps $X$),
for a fixed value of $\theta$

$(x_i, y_i)$
- For least square problems, we are maximizing

$$
\ell(w) = \prod_{i=1}^{m}{p(y^i|x^i;w)} = \prod_{i=1}^{m}{\frac{1}{\sqrt{2\pi\sigma}}e^{-\frac{(y^i-w^Tx^i)^2}{2\sigma^2}}}
$$

- For logistic regression, we are maximizing

$$
\ell(\theta) = \prod_{i=1}^{m}{p(y^i|x^i;\theta)} = \prod_{i=1}^{m}{(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}}
$$
    
    - We then take the log of this function since it make the life easier.

- For naive bayes of binary output, we are actually maximizing the joint distribution of data, since the our probability distribution depends on both features and labels.

$$
\begin{gather*}
\ell(\theta,\mu_{0},\mu_{1},\Sigma) = log\prod_{i=1}^{m}{p(x^i,y^i;\theta,\mu_0,\mu_1,\Sigma)} \\
y\sim{Bernoulli(\theta)} \\
x\mid y=0 \sim \mathcal{N}(\mu_{0},\,\Sigma) \\
x\mid y=1 \sim \mathcal{N}(\mu_{1},\,\Sigma)
\end{gather*}
$$

### Derivation of Naive Bayes
I'll skip this part for now

### Variations of Naive Bayes
- Multi-variate Bernoulli
    - Good for binary feature vector
    - Good for modeling text classification, word vector model. [0, 1, 0, ...]
- Multinomial Model
    - Quite similiar to Bernoulli Naive bayese, but the distribution is multinomial. For large training set, the accuracy is usually better than the previous one. However this is from empirical experience, in fact the feature selection part largely determines the performance.
    - Good for modeling text classification as well, word event model.
- LDA
    - Continous features, covarience matrix doesn't change
- QDA
    - Continous features, covarience matrix change depending on prior classes

### Something to notice about applying Bayes in text classification
- Usually the size of the samples is outnumbered by the number of features when doing text classification. Namely, we have n >> m. This may lead to generalization error which means the model might overfit. 
- Andrew Ng has done some research to regulate this approach to make it efficient in text classification. We won't need to go into the details but it's good to know.