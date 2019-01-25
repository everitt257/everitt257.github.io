---


---

<h1 id="boostrap-sampling--bagging-summary">Boostrap sampling &amp; Bagging Summary</h1>
<ul>
<li>bootstrap sampling {may sample the same sample}</li>
<li>bootstrap sampling under bagging framework {take multiple samples to train individual classifier}</li>
<li>weak classifier put under bagging framework</li>
<li>everything combined --&gt; ensemble learning</li>
<li>weighted weak classifier, training sampling weighting --&gt; adaboosting
<ul>
<li>adaboosting induction
<ul>
<li>general weighted function for prediction</li>
<li>make use of exponetial function <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><msup><mi>e</mi><mrow><mo>−</mo><mi>Y</mi><mo>⋅</mo><mi>f</mi><mo>(</mo><mi>X</mi><mo>)</mo></mrow></msup></mrow><annotation encoding="application/x-tex">e^{-Y\cdot f(X)}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.888em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathit">e</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.888em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">−</span><span class="mord mathit mtight" style="margin-right: 0.22222em;">Y</span><span class="mbin mtight">⋅</span><span class="mord mathit mtight" style="margin-right: 0.10764em;">f</span><span class="mopen mtight">(</span><span class="mord mathit mtight" style="margin-right: 0.07847em;">X</span><span class="mclose mtight">)</span></span></span></span></span></span></span></span></span></span></span></span></span> for comparing similarity {reason for using this is because it has better performance than l2 loss in classification?}</li>
<li>link for <a href="https://mp.weixin.qq.com/s?__biz=MzU4MjQ3MDkwNA==&amp;mid=2247486478&amp;idx=1&amp;sn=8557d1ffbd2bc11027e642cc0a36f8ef&amp;chksm=fdb69199cac1188ff006b7c4bdfcd17f15f521b759081813627be3b5d13715d7c41fccec3a3f&amp;scene=21#wechat_redirect">induction</a></li>
<li>application: face detection, before nn was fully applicable, make use of Haar features</li>
</ul>
</li>
</ul>
</li>
<li>weak classifier made up with decision tree --&gt; random forest
<ul>
<li>decision tree
<ul>
<li>core idea: search all feature space to find one feature that achieves maximum information gain.</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>max</mi><mo>⁡</mo><msub><mi>E</mi><mrow><mi>g</mi><mi>a</mi><mi>i</mi><mi>n</mi></mrow></msub><mo>=</mo><msub><mi>max</mi><mo>⁡</mo><mrow><mi>f</mi><mi>e</mi><mi>a</mi><mi>t</mi><mi>u</mi><mi>r</mi><mi>e</mi><mi>s</mi></mrow></msub><mo>(</mo><msub><mi>E</mi><mrow><mi>o</mi><mi>r</mi><mi>i</mi><mi>g</mi><mi>i</mi><mi>n</mi><mi>a</mi><mi>l</mi></mrow></msub><mo>−</mo><msub><mi>E</mi><mrow><mi>s</mi><mi>p</mi><mi>l</mi><mi>i</mi><mi>t</mi></mrow></msub><mo>)</mo></mrow><annotation encoding="application/x-tex">\max E_{gain} = \max_{features} (E_{original} - E_{split})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.969438em; vertical-align: -0.286108em;"></span><span class="mop">max</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.05764em;">E</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.05764em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.03588em;">g</span><span class="mord mathit mtight">a</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight">n</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mop"><span class="mop">max</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight" style="margin-right: 0.10764em;">f</span><span class="mord mathit mtight">e</span><span class="mord mathit mtight">a</span><span class="mord mathit mtight">t</span><span class="mord mathit mtight">u</span><span class="mord mathit mtight" style="margin-right: 0.02778em;">r</span><span class="mord mathit mtight">e</span><span class="mord mathit mtight">s</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mopen">(</span><span class="mord"><span class="mord mathit" style="margin-right: 0.05764em;">E</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: -0.05764em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">o</span><span class="mord mathit mtight" style="margin-right: 0.02778em;">r</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight" style="margin-right: 0.03588em;">g</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight">n</span><span class="mord mathit mtight">a</span><span class="mord mathit mtight" style="margin-right: 0.01968em;">l</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">−</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 1.03611em; vertical-align: -0.286108em;"></span><span class="mord"><span class="mord mathit" style="margin-right: 0.05764em;">E</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.336108em;"><span class="" style="top: -2.55em; margin-left: -0.05764em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathit mtight">s</span><span class="mord mathit mtight">p</span><span class="mord mathit mtight" style="margin-right: 0.01968em;">l</span><span class="mord mathit mtight">i</span><span class="mord mathit mtight">t</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.286108em;"><span class=""></span></span></span></span></span></span><span class="mclose">)</span></span></span></span></span> in another word, maximizes the entropy gain is the same as minimizes the impurity.</li>
<li>classification {measures with entropy gain, gini purity, miss-classification}</li>
<li>regression {measures with  l2 loss, mapping of piecewise constant function}<br>
<a href="https://postimg.cc/D4gZYzpR"><img src="https://i.postimg.cc/PNGwJpWf/image.png" alt="image.png"></a><br>
<em>since it is piecewise constant function, if we take a decesion tree and seperate it small enough, in theory it can simulate any non-linear function.</em></li>
</ul>
</li>
<li>drawbacks of decision tree
<ul>
<li>as long as the depth of the tree is deep enough, we can achieve very high precision in the test set. However when the feature dimension is too high, the “curse of dimension” may happen and the model will overfit.</li>
<li>complex trimming technique, it’s like tuning hyper-parameters. Many methods exist, the common one used in classifiying is Cost-Complexity Pruning (CCP).</li>
</ul>
</li>
<li>random forest made up with multiple decision tree
<ul>
<li>each tree is a weak classifier, with merely 50% accucray is enough</li>
<li>each tree is made with random sample</li>
<li>also the feature for composing the tree is randomly selected</li>
</ul>
</li>
<li>multiple decision tree compared with single decision tree effectively reduces the variance.</li>
</ul>
</li>
<li>xgboost?</li>
</ul>

