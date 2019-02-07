---
layout: post
category: blog
title: Decision Tree Variants
tags: [machine learning, trees]
---

In this post we dicuss different decision trees.

Two types of decision tree exists.
- Classification Tree.
	- The simplest to exist. See Bootstrap and Bagging for details.
- Regression Tree
	- Make use of variance as a measure for splitting.
- Adaboost with decision tree
	- same as random forest except there the weights of each classifier and sample are re-calculated during each iteration
- GBDT (Gradient Boost Decision Tree) (Mart) (Multiple Additive Regression Tree)
	- think of it as residual net
	- two kinds of implementation
		- new tree trained gradient based
		- new tree trained difference based
	- loop procedure: initial setup $y_{label} = y_{actual}$
		- $y_{n} = y_{label} - y_{1}^{n-1}$, where $y_1^{n-1}$ represent an ensemble prediction of all previous trees. And $y_n$ is the new tree learned from either the differene or gradient
		- $y_1^n = y_1^{n-1} + y_n$, the new tree is then added to the ensemble
		- $y_{label} = \text{difference or negative gradient}$ 
	- shrinkage version
		- replace step 2 in the loop with $y_1^n = y_1^{n-1} + \text{step} \cdot y_n$, everything else is the same.
		- require more computational resource
		- prevent overfitting. No theory to prove it though.
	- other variant: stochastic sampling of features and bootstrap sampling.
	- [Reference](https://blog.csdn.net/suranxu007/article/details/49910323)
- XGBOOST
	- make use of 2nd Taylor series
	- 2nd Taylor series reduces the iteration process therefore hasten the training process
	- added 2 regulations to prevent the tree from overfitting
	- [![kltaUH.png](https://s2.ax1x.com/2019/01/30/kltaUH.png)](https://imgchr.com/i/kltaUH) the loss function
	- ![kltOIJ.png](https://s2.ax1x.com/2019/01/30/kltOIJ.png) the regulation function
	- where $\gamma$ and $\lambda$ is two hyperparameters, the $\gamma$ parameter is a threshold value. And $\lambda$ is a smoothing factor.
	- [Reference - Usuage](https://blog.csdn.net/qunnie_yi/article/details/80129857)
	- [Reference - Induction](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247488624&idx=1&sn=078f5440b3bae6bd1699afe65995d21f&chksm=fdb689e7cac100f1ff758cbd909cfed6863aa078583c2b1b5b6f277d51d6852d9bac4e681394&mpshare=1&scene=1&srcid=&pass_ticket=hkqcq4hgS2KK5LbCxtVkFhphZJgo%2bVpKa974a2nljT1JVjS2/LpWDI3O45r8jerN#rd)
	- **Features I don't understand**
		- how parallel process is implemented in XGboost
		- Its relationship to GBDT
		- How it handles missing features
		- If 2nd order derivative is used, why not 3rd order?