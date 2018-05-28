---
layout: post
title: Various Length RNN Applied in Blog Data
category: blog
tags: algorithm
---
This is an implementation of various-length RNN with single layer GRU model. Due to time limitation, it only showcases the basics of RNN model.


```python
import pandas as pd, numpy as np, tensorflow as tf
import blogs_data #available at https://github`.com/spitis/blogs_data
```

    /home/everitt257/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



```python
df = blogs_data.loadBlogs().sample(frac=1).reset_index(drop=True)
vocab, reverse_vocab = blogs_data.loadVocab()
train_len, test_len = np.floor(len(df)*0.8), np.floor(len(df)*0.2)
train, test = df.loc[:train_len-1], df.loc[train_len:train_len + test_len]
df = None
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>post_id</th>
      <th>gender</th>
      <th>age_bracket</th>
      <th>string</th>
      <th>as_numbers</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>144744</td>
      <td>1</td>
      <td>0</td>
      <td>we listened to this creepy music we all were s...</td>
      <td>[32, 1968, 5, 29, 3623, 344, 32, 37, 88, 942, ...</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>84957</td>
      <td>1</td>
      <td>1</td>
      <td>when a person &lt;UNK&gt; , the throat closes to pre...</td>
      <td>[56, 7, 211, 0, 1, 4, 2379, 8457, 5, 3071, 443...</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>134300</td>
      <td>1</td>
      <td>0</td>
      <td>&lt;UNK&gt; ... guess those that stayed back in clas...</td>
      <td>[0, 24, 228, 161, 9, 1024, 93, 11, 320, 66, 64...</td>
      <td>14</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11751</td>
      <td>1</td>
      <td>0</td>
      <td>speaking of money i got my atm card fixed today !</td>
      <td>[973, 8, 314, 3, 89, 13, 7210, 983, 2062, 119,...</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>126685</td>
      <td>0</td>
      <td>1</td>
      <td>as of now , around &lt;#&gt; hours from her phone ca...</td>
      <td>[38, 8, 68, 1, 146, 12, 309, 57, 61, 397, 260,...</td>
      <td>18</td>
    </tr>
  </tbody>
</table>
</div>




```python
class SimpleDataIterator():
    def __init__(self, df):
        self.df = df
        self.size = len(self.df)
        self.epochs = 0
        self.shuffle()
    
    def shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True) # not sure why reset_index is used at here
        self.cursor = 0
    
    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.loc[self.cursor: self.cursor+n-1]
        self.cursor += n
        return res['as_numbers'], res['gender']*3 + res['age_bracket'], res['length']
```


```python
data = SimpleDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])
```

    Input sequences
     0    [0, 1, 0, 49, 0, 50, 200, 9, 465, 19, 2514, 13...
    1    [723, 52, 153, 30, 771, 33, 2145, 33, 4073, 79...
    2    [6, 1863, 14, 13, 2678, 2482, 32, 97, 843, 0, ...
    Name: as_numbers, dtype: object
    
    Target values
     0    0
    1    3
    2    3
    dtype: int64
    
    Sequence lengths
     0    16
    1    11
    2    26
    Name: length, dtype: int64


## Problem
The three sequences are of different length.
### Solution
Pad different length sequences into the same length so the can be fit into the same tensor.


```python
class PaddedDataIterator(SimpleDataIterator):
    def next_batch(self, n):
        if self.cursor+n > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.df.loc[self.cursor: self.cursor+n-1]
        self.cursor += n
    
        # Pad the various sequences with zeroes to make them the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]
        
        return x, res['gender']*3 + res['age_bracket'], res['length']
```


```python
data = PaddedDataIterator(train)
d = data.next_batch(3)
print('Input sequences\n', d[0], end='\n\n')
print('Target values\n', d[1], end='\n\n')
print('Sequence lengths\n', d[2])
```

    Input sequences
     [[ 286    1 5364   42  382  153   80   15  743  116    7 2925  742    1
        10   22   34   40   36  229   15    4 1819    8 2925   50]
     [   6   65   19  289 6197   42 5973    5  771    6 2708  151   25    0
         0    0    0    0    0    0    0    0    0    0    0    0]
     [ 386    0  390 1422   13  213 1079   16   61  382   54  474   40   12
         6    3  429 1213 5687    0 8235   14  800   25    0    0]]
    
    Target values
     0    3
    1    3
    2    0
    dtype: int64
    
    Sequence lengths
     0    26
    1    13
    2    24
    Name: length, dtype: int64


## Basic model for sequence classification
Have the model guess the outcome at the very last step


```python
def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

def build_graph(
    vocab_size = len(vocab),
    state_size = 64,
    batch_size = 256,
    num_classes = 6):
    
    reset_graph()
    
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, None]
    seqlen = tf.placeholder(tf.int32, [batch_size]) #[batch_size]
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder(tf.float32, [])
    
    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x) #[batch_size, None, state_size]
    
    # RNN with GRU
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    # Easier this way: init_state = cell.zero_state(batch_size, tf.float32) 
    init_state = tf.get_variable('init', [1, state_size]) #[1, state_size]
    init_state = tf.tile(init_state, [batch_size, 1]) # replicate 256 piece of it
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=seqlen, initial_state=init_state)
    
    # rnn_outputs = [batch_size, None, state_size], final_state = [batch_size, 1, state_size]
    # It's actually a single layer gru with dropout
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    
    # obtain the last relevant outputs, shape=(256, 64), note taht seqlen is of different sizes
    last_rnn_output = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), seqlen-1], axis=1))
    
    # adds a softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(last_rnn_output, W) + b
    preds = tf.nn.softmax(logits) # shape=(batch_size, num_classes)
    correct = tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y) # shape=(batch_size,)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }

def train_graph(graph, batch_size = 256, num_epochs = 10, iterator = PaddedDataIterator, dropout = 0.6):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tr = iterator(train)
        te = iterator(test)
        
        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        while current_epoch < num_epochs:
            step += 1
            batch = tr.next_batch(batch_size)
            feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: dropout}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_
            
            if tr.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                # reset for evaluation
                step, accuracy = 0, 0
                # eval test set
                te_epoch = te.epochs
                while te.epochs == te_epoch:
                    step += 1
                    batch = te.next_batch(batch_size)
                    feed = {g['x']: batch[0], g['y']: batch[1], g['seqlen']: batch[2], g['dropout']: 1.0}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_
                    
                te_losses.append(accuracy / step)
                # reset after the evaluation, to continue to evaluate training accuracy on next epoch
                step, accuracy = 0, 0
                print("Accuracy after epoch", current_epoch, " - tr:", tr_losses[-1], "- te:", te_losses[-1])
    
    return tr_losses, te_losses
```

### Basic test


```python
g = build_graph()
tr_losses, te_losses = train_graph(g)
```

    Accuracy after epoch 1  - tr: 0.3160342651213897 - te: 0.34959466527196653
    Accuracy after epoch 2  - tr: 0.354831745865606 - te: 0.35830192629815744
    Accuracy after epoch 3  - tr: 0.3617226750575675 - te: 0.35916234819932996
    Accuracy after epoch 4  - tr: 0.36376125183169356 - te: 0.35978394577051925
    Accuracy after epoch 5  - tr: 0.3654105937303747 - te: 0.3596530831239531
    Accuracy after epoch 6  - tr: 0.36659465276324055 - te: 0.36046443153266333
    Accuracy after epoch 7  - tr: 0.36802484561440235 - te: 0.3609159076633166
    Accuracy after epoch 8  - tr: 0.36934382850115133 - te: 0.361710898241206
    Accuracy after epoch 9  - tr: 0.37092448189240107 - te: 0.36232922424623115
    Accuracy after epoch 10  - tr: 0.37257627695206197 - te: 0.36191373534338356


## Improving with bucketing
Since there are a lot of zeros in the sequences, let's calculate the average padding of zeros.


```python
tr = PaddedDataIterator(train)
padding = 0
for i in range(100):
    lengths = tr.next_batch(256)[2].values
    max_len = max(lengths)
    padding += np.sum(max_len - lengths)
print("Average padding / batch:", padding/100)
```

    Average padding / batch: 3291.11



```python
class BucketDataIterator():
    def __init__(self, df, num_buckets = 5):
        df = df.sort_values('length').reset_index(drop=True)
        self.size = len(df)/num_buckets # each bucket's size
        self.dfs = []
        for bucket in range(num_buckets):
            self.dfs.append(df.loc[bucket*self.size: (bucket+1)*self.size - 1])
            
        self.num_buckets = num_buckets
        
        self.cursor = np.array([0] * num_buckets)
        self.shuffle()
        self.epochs = 0
        
    def shuffle(self):
        for i in range(self.num_buckets):
            # sorts dataframe by sequence length, but keeps it random within the same length
            self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
            self.cursor[i] = 0
            
    def next_batch(self, n):
        # this part acts as overwatch for batch reaching the end of a small df
        if np.any(self.cursor+n+1 > self.size):
            self.epochs += 1
            self.shuffle()
        
        i = np.random.randint(0, self.num_buckets)
        res = self.dfs[i].loc[self.cursor[i]: self.cursor[i]+n-1]
        self.cursor[i] += n
        # Pad sequences with 0s so they are all the same length
        maxlen = max(res['length'])
        x = np.zeros([n, maxlen], dtype=np.int32)
        for i, x_i in enumerate(x):
            x_i[:res['length'].values[i]] = res['as_numbers'].values[i]

        return x, res['gender']*3 + res['age_bracket'], res['length']
```


```python
tr = BucketDataIterator(df=train, num_buckets=5)
padding = 0
for i in range(100):
    lengths = tr.next_batch(256)[2].values
    max_len = max(lengths)
    padding += np.sum(max_len - lengths)
print("Average padding / batch:", padding/100)
```

    Average padding / batch: 583.76



```python
from time import time
g = build_graph()
t = time()
tr_losses, te_losses = train_graph(g, num_epochs=1, iterator=PaddedDataIterator)
print("Total time for 1 epoch with PaddedDataIterator:", time() - t)
```

    WARNING:tensorflow:From /home/everitt257/anaconda3/lib/python3.6/site-packages/tensorflow/contrib/learn/python/learn/datasets/base.py:198: retry (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use the retry module or similar alternatives.
    Accuracy after epoch 1  - tr: 0.3123986239012139 - te: 0.3490258891213389
    Total time for 1 epoch with PaddedDataIterator: 380.0432941913605



```python
g = build_graph()
t = time()
tr_losses, te_losses = train_graph(g, num_epochs=1, iterator=BucketDataIterator)
print("Total time for 1 epoch with BucketedDataIterator:", time() - t)
```

    Accuracy after epoch 1  - tr: 0.31257560483870966 - te: 0.34867950074701193
    Total time for 1 epoch with BucketedDataIterator: 316.3720586299896


## Basic sequence to sequence learning
Have the model guess at every step!


```python
def build_seq2seq_graph(
    vocab_size = len(vocab),
    state_size = 64,
    batch_size = 256,
    num_classes = 6):
    
    reset_graph()
    
    # Placeholders
    x = tf.placeholder(tf.int32, [batch_size, None]) # [batch_size, num_steps]
    seqlen = tf.placeholder(tf.int32, [batch_size])
    y = tf.placeholder(tf.int32, [batch_size])
    keep_prob = tf.placeholder(tf.float32, [])
    
    # Tile the target indices
    y_ = tf.tile(tf.expand_dims(y, 1), [1, tf.shape(x)[1]]) # [batch_size, num_steps]
    
    lower_triangular_ones = tf.constant(np.tril(np.ones([30,30])),dtype=tf.float32) # since 30 is of maximum length
    # tf.gather returns [batch_size, 30], tf.slice returns [batch_size, max(length of current batch)]
    seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, seqlen-1),\
                           [0, 0], [batch_size, tf.reduce_max(seqlen)])
    
    # Embedding layer
    embeddings = tf.get_variable('embedding_matrix', [vocab_size, state_size])
    rnn_inputs = tf.nn.embedding_lookup(embeddings, x)
    
    # RNN
    cell = tf.nn.rnn_cell.GRUCell(state_size)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, 
                                                 sequence_length=seqlen, 
                                                 initial_state=init_state)
    
    # Adds dropout
    rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob)
    
    # Reshape rnn_outputs and y
    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    y_reshaped = tf.reshape(y_, [-1])
    
    # Softmax layer
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(rnn_outputs, W) + b
    
    preds = tf.nn.softmax(logits)
    
    # Calculate 
    correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds,1),tf.int32), y_reshaped),tf.int32) *\
                tf.cast(tf.reshape(seqlen_mask, [-1]),tf.int32)
        
    # To calculate accuracy we want to divide by the number of non-padded time-steps,
    # rather than taking the mean
    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(seqlen, tf.float32))
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = loss * tf.reshape(seqlen_mask, [-1])
    
    # To calculate average loss, we need to divide by number of non-padded time-steps,
    # rather than taking the mean
    loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return {
        'x': x,
        'seqlen': seqlen,
        'y': y,
        'dropout': keep_prob,
        'loss': loss,
        'ts': train_step,
        'preds': preds,
        'accuracy': accuracy
    }
```

## Test seq2seq
Should behave worst than sequence classification.


```python
g = build_seq2seq_graph()
tr_losses, te_losses = train_graph(g, iterator=BucketDataIterator)
```

    Accuracy after epoch 1  - tr: 0.29417641936604116 - te: 0.3165940021659805
    Accuracy after epoch 2  - tr: 0.32053691482279945 - te: 0.32140361183230964
    Accuracy after epoch 3  - tr: 0.3248470337743139 - te: 0.3230788176369774
    Accuracy after epoch 4  - tr: 0.32654755128969826 - te: 0.32458225125352463
    Accuracy after epoch 5  - tr: 0.3278110629056041 - te: 0.3248658520595809
    Accuracy after epoch 6  - tr: 0.329185960110063 - te: 0.32527022973785763
    Accuracy after epoch 7  - tr: 0.33046067700309273 - te: 0.3260545640350034
    Accuracy after epoch 8  - tr: 0.33134092020356276 - te: 0.32620091709176663
    Accuracy after epoch 9  - tr: 0.332484958309292 - te: 0.32587733625728177
    Accuracy after epoch 10  - tr: 0.3326295395103957 - te: 0.3262810798193513
