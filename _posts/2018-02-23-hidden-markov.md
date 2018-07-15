---
layout: post
title: Hidden Markov Chain Model
tags: algorithm
category: blog
---
# Hidden Markov Model
The whole reason I'm recording this is because RF uses some of properties of Hidden Markov Model or the Kalman-filter model to estimate the hidden state. *Notice the difference between HMM and KF is that one is discrete time spaced and the other continuous time spaced*.
## The Bayesian network
The Bayesian network is merely a way to graphically think conditional probability.

![](http://om1hdizoc.bkt.clouddn.com/18-1-20/61211680.jpg)

## The D-seperateion:
The D-seperation is used for trackable inference.
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/14462958.jpg)
Also notice that the D-seperation can be used in Bayesian network as well. Check video link here:
[D seperation](https://www.youtube.com/watch?v=zJIK5uOyJi0&list=PLD0F06AA0D2E8FFBA&index=92)
## The Forward-backward procedure
The algorithm that reveals $p(z_k \mid x)$ uses the forward and backward algorithm.
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/21967713.jpg)
By using the idea of d-seperation, we can reduce $p(z_k \mid x)$ into something simpler as above. 
> In one online discussion, someone called this term $p(z_k \mid x)$ as filtering.[[1]](https://stats.stackexchange.com/questions/183118/difference-between-hidden-markov-models-and-particle-filter-and-kalman-filter)

The first term $p(x_{k+1:n} \mid z_k)$ is the so called for **backward procedure**. The second term $p(z_k, x_{1:k})$ is the **forward procedure**.
### The forward algorithm
The step-back trick:
$$p(z_k, x_{1:k}) = \sum^m_{z_{k-1} = 1} p(z_k, z_{k-1}, x_{1:k})$$
By the tree structure and the d-seperation principle and with this trick in mind:
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/19958631.jpg)
The result is thus:
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/87529527.jpg)
- The first equality comes from the stepback trick.
- The second equality comes from the chain rule of probability **and** the D-seperation principle.

we have the following equation:
$$p(z_k, x_{1:k}) = \sum_{z_{k-1}} p(x_k \mid z_k) p(z_k \mid z_{k-1}) p(z_{k-1}, x_{1:k-1})$$
The first term is the **emission probability**, the second term is the **transition probability**. As you can see, this is a **recursive structure**. Normally, the emission and transition are given. They correspond to A and C.

#### Application: finding observed sequence probability
Also, the forward pass can be summed up to calculate the oberserved sequence probability. There's an alternative(in Andrew's note) which is more computationally expensive, thus we use this approach instead.
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/75267330.jpg)
### The backward algorithm
Similar to the forward algorithm, the backward algorithm also make use of the recursive structure, which is the heart of dynamic programming.
![](http://om1hdizoc.bkt.clouddn.com/18-1-20/5769210.jpg)
## The Viterbi algorithm
One of the most common queries of a Hidden Markov Model is to ask what was the most likely series of states $\vec{z} \in S^T$ given an observed series of outputs $\vec{x} \in V^T$. One potential application of viterbi algorithm can be used in hand writing recognition. 
### The deduction proof
The deduction is as follows:

Given: $x = x_{1:n}$ and $z = z_{1:n}$, the goal is to compute $$z^* = \text{arg max}_z p(z \mid x)$$
Notice that
$$\text{arg max}_{z_{1:n}} p(z \mid x) = \text{arg max}_{z_{1:n}} p(z,x)$$
also let
$$u_k(z_k) = max_{z_{1:k-1}}p(z_{1:k},x_{1:k})$$
along with the fact that
![](http://om1hdizoc.bkt.clouddn.com/18-1-21/29244779.jpg)
also with help of chain rule of probability and the D-seperation in our network. We can come to another recursion structure:
 ![](http://om1hdizoc.bkt.clouddn.com/18-1-21/70736219.jpg)
 This essentially say that in order to reveal the hidden sequence of state. We need to calculate every possible sequence's probability value recursively. In another word, the $u_k(z_k)$ term means the optimal step to take at time step k. If you know the end hidden state in advance then you can iterate all path's probability value to find the optimal choice. However if k at current time step is just another intermediate step, one would need to calculate all possible hidden state's optimal choices.  Note that in this setting we assume $A$ the **state transition probability matrix** and $B$ the **output emission probability matrix** is known. Normally one would need to run ML to estimate these parameters.
 
## The EM for HMM
The  final question would be, given a set of observations, what are the values of the state transition probabilities A and the output emission probabilities B that make the data most likely. 

- The Naive application of EM to HMMs
![](http://om1hdizoc.bkt.clouddn.com/18-1-21/10373446.jpg)
The derivation of the EM make use of Markov assumption (which can be proved by d-seperation in Bayesian network). Also it make use of Lagrange multiplier and Forward-Backward algorithms and a bunch of things to come up with the following variant of EM for HMM.
- The Forward-Backward algorithm for HMM parameter learning
![](http://om1hdizoc.bkt.clouddn.com/18-1-21/54629756.jpg)
In some sense, $\gamma_t(i,j)$ can be computed statistically.  It is proportional to the probability of transitioning between state $s_i$ and $s_j$ at time $t$ given all of our observations $\vec{x}$.
 > Like many applications of EM, parameter learning for HMMs is a non-convex problem with many local maxima. EM will converge to a maximum based on its initial parameters, so multiple runs might be in order. Also, it is often important to smooth the probability distributions represented by A and B so that no transition or emission is assigned 0 probability. [[2]](http://cs229.stanford.edu/section/cs229-hmm.pdf)
 >  - The last line refer to the log exp trick.
 
## Relationship to KF, PF, and other time series estimator model
|State|Dynamics|Noise|Generalization|Cost|Model|
|:--:|:--|:--|:--|:--|:--:|
|Continuous/Discrete|Linear|Normal|Only Linear System|Cheapest|KF|
|Continuous/Discrete|Linear/None-linear|Normal|Depending on the order of Taylor series, the 2rd is arguably a better choice than UKF|Cheap|EKF|
|Continuous/Discrete|Linear/None-linear|Normal|Able to model non-linear system better than 1st order EKF|Less Expensive|UKF|
|Continuous/Discrete|Linear/None-linear|Any|Strong, should be able to model any dynamics and any distribution|Expensive, difficult to tune as well|PF|
|Discrete|Difficult to compare, the above filters uses state space model, while HMM uses probabilistic model|No assumption of Gaussian or linear what so ever|Time series of data or temporal data|Uses the forward-backward algorithm which is proportional to $O(\|S\|*T )$|HMM|

Some will say that KF or variants of KF is a special case of HMM. I hold my doubts about that since I haven't read related papers yet.