# Boostrap sampling & Bagging Summary

- bootstrap sampling {may sample the same sample}
- bootstrap sampling under bagging framework {take multiple samples to train individual classifier}
- weak classifier put under bagging framework
- everything combined --> ensemble learning
- weighted weak classifier, training sampling weighting --> adaboosting
	- adaboosting induction
		- general weighted function for prediction
		- make use of exponetial function $e^{-Y\cdot f(X)}$ for comparing similarity {reason for using this is because it has better performance than l2 loss in classification?}
		- link for [induction](https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&mid=2247486478&idx=1&sn=8557d1ffbd2bc11027e642cc0a36f8ef&chksm=fdb69199cac1188ff006b7c4bdfcd17f15f521b759081813627be3b5d13715d7c41fccec3a3f&scene=21#wechat_redirect)
		- application: face detection, before nn was fully applicable, make use of Haar features
- weak classifier made up with decision tree --> random forest
	- decision tree
		- core idea: search all feature space to find one feature that achieves maximum information gain.
		- $\max E_{gain} = \max_{features} (E_{original} - E_{split})$ in another word, maximizes the entropy gain is the same as minimizes the impurity.
		- classification {measures with entropy gain, gini purity, miss-classification}
		- regression {measures with  l2 loss, mapping of piecewise constant function}
		[![image.png](https://i.postimg.cc/PNGwJpWf/image.png)](https://postimg.cc/D4gZYzpR)
		*since it is piecewise constant function, if we take a decesion tree and seperate it small enough, in theory it can simulate any non-linear function.*
	- drawbacks of decision tree
		- as long as the depth of the tree is deep enough, we can achieve very high precision in the test set. However when the feature dimension is too high, the "curse of dimension" may happen and the model will overfit.
		- complex trimming technique, it's like tuning hyper-parameters. Many methods exist, the common one used in classifiying is Cost-Complexity Pruning (CCP). 
	- random forest made up with multiple decision tree
		- each tree is a weak classifier, with merely 50% accucray is enough
		- each tree is made with random sample
		- also the feature for composing the tree is randomly selected 
	- multiple decision tree compared with single decision tree effectively reduces the variance.
- xgboost?



