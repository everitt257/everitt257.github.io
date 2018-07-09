---
layout: post
title: Discussion of generation models, sequential generation
tags: [deep learning, algorithm]
category: blog
---

Previously I've read about R2RT's post regarding sequential generation and discrete embeddings. I thought I'll write something to remind myself the details of this post and to compare it with other generation models.

Discrete embedding can prevent overfitting as we've seen it in my other post. It can also communicate fuzzy ideas with concrete symbols. This means (1) we can create a language model over the embeddings, which give us give RNN-based generation of internal embeddings, and (2) index sub-parts of the embeddings, whichi gives us access to the search techniques that go beyong cosine similarity, such as phrase search.

We'll use MNIST dataset for illustration. The ultimate goal is as described: given a sequence of imcomplete features, can we we find subsequences of consecutive digits that have these features?

{:.mycenter}
![](http://om1hdizoc.bkt.clouddn.com/18-6-22/1743208.jpg)

Notice the original's post's solution to this question is not perfect. You might want to view the original post at [here](https://r2rt.com/deconstruction-with-discrete-embeddings.html).

### Embeddings
Why discrete embeddings not real embeddings like w2v? While real embeddings may capture more details regarding the dataset, such as width, heights, angles etc, the discrete embeddings allows user to apply explicit reasoning and algoirhtms over the data and it helps with overfitting. But we can always use both, for example, a mixture of both real and discrete embeddings during our training.

### Autoencoder
The original post talks about building an autoencoder with discrete embeddings with one digit in the latent variable unused. We are just going to conclude that discrete embeddings are sufficient for reconstructing the original post and 560 zeros and 80 ones are sufficient in communicating during the reconstruction.

### Sequential Generator
The real beauty of this post comes when it tries to reconstruct images with RNN. To illustrate this, we will show the original code.

```python
def imgs_to_indices(imgs):
    embs = sess.run(g['embedding'], feed_dict={g['x']: imgs, g['stochastic']: False}) #[n, 80, 8]
    idx = np.argmax(embs, axis=2) #[n, 80]
    res = []
    for img in range(len(imgs)):
        neuron_perm = np.random.permutation(list(range(80))) #order of neurons we will present
        res.append(idx[img][neuron_perm] + neuron_perm * 8)
    return np.array(res)

def gen_random_neuron_batch(n):
    x, _ = mnist.train.next_batch(n) # [n, 784]
    res = imgs_to_indices(x)
    return res[:,:-1], res[:,1:]
```
As explained in the original post, there are no hierarchical structure regarding the order of how of how we query our discrete embeddings. Therefore we randomize the order and produce such random neuron batch. Taking a closer look this code:

```python
neuron_perm = np.random.permutation(list(range(80))) #order of neurons we will present
res.append(idx[img][neuron_perm] + neuron_perm * 8)
```

First we are adding randomness to the order. Secondly we are preserving the indexes of the neuron. Without the digit 8, we'll have duplicates and the order will be messed up. Therefore we must use 640 to index each our neuron in each example.

Another thing to notice is that since randomness is added, we are essentially transforming the discrete embeddings trained in the autoencoder to randomized **sequential** embeddings. And then applied to RNN. This is essential since two model (autoencoder and RNN) uses two different kind of embeddings.

Supply with mermaid graph here:

<style>
.mycenter {
    text-align:center;
}
</style>