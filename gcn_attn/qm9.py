#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
import numpy as np


# In[21]:


def data_parse(record):
    features = {
        'N': tf.io.FixedLenFeature([], tf.int64),
        'labels': tf.io.FixedLenFeature([16], tf.float32),
        'elements': tf.io.VarLenFeature(tf.int64),
        'coords': tf.io.VarLenFeature(tf.float32),
    }
    parsed_features = tf.io.parse_single_example(
        serialized=record, features=features)
    coords = tf.reshape(tf.sparse.to_dense(parsed_features['coords'], default_value=0), [-1, 4])
    elements = tf.sparse.to_dense(parsed_features['elements'], default_value=0)    
    return (elements, coords), parsed_features['labels']
record_file = 'qm9.tfrecords'
data = tf.data.TFRecordDataset(
    record_file, compression_type='GZIP').map(data_parse)


# In[66]:


{'C': 6, 'H': 1, 'O': 8, 'N': 7, 'F': 9}

def make_graph(e, x):
    e = e.numpy()
    x = x.numpy()
    r = x[:, :3]
    r2 = np.sum((r - r[:, np.newaxis, :])**2, axis=-1)
    edges = np.where(r2 != 0, 1 / r2, 0.)
    nodes = np.zeros((len(e), 9))
    nodes[np.arange(len(e)), e - 1] = 1
    return nodes, edges

def get_label(y):
    return y.numpy()[13]


# In[69]:


for d in data:
    (e, x), y = d
    nodes, edges = make_graph(e, x)
    label = get_label(y)
    print(nodes, edges, label)
    break


# In[28]:


import jax.numpy as jnp
import jax.experimental.optimizers as optimizers
import jax


# In[91]:


r = np.repeat(nodes[np.newaxis,...], nodes.shape[0], axis=0) @ np.ones((9, 10))


# In[100]:


def gcn_layer(nodes, edges, w):
    messages = jnp.dot(jnp.repeat(nodes[jnp.newaxis,...], nodes.shape[0], axis=0),  w) * edges[...,jnp.newaxis]
    net_message = jnp.mean(messages, axis=1)
    out = jax.nn.relu(net_message) + nodes @ w
    return out, edges


# In[105]:


embedding_dimension = 3
element_embeddings = np.random.normal(size=(9, 3))
embedded_nodes = nodes @ element_embeddings
w1 = np.random.normal(size=(3, 10))
w2 = np.random.normal(size=(3, 10))
n,e = gcn_layer(embedded_nodes, edges, w1)
n,e = gcn_layer(embedded_nodes, edges, w2)
print(n, e)


# In[ ]:




