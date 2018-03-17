import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter

import pickle as pkl

text = []
with open('corpus_0307_1.txt','r') as f:
    for line in f.readlines():
        line = line.strip('\n')
        text.extend(line.split(" "))
print(len(text))

words_count = Counter(text)
print(len(words_count))
words = [w for w in text if words_count[w]>=5]
print(set(words))

vocab_dict = set(words)
vocab_to_int = {w:c for c,w in enumerate(vocab_dict)}
int_to_vocab = {c:w for c,w in enumerate(vocab_dict)}
#print(vocab_to_int)
print("total words: {}".format(len(words)))
print("unique words: {}".format(len(vocab_dict)))
int_words = [vocab_to_int[w] for w in words]
train_words = int_words

def get_target(words,idx,window_size = 5):
    target_window = np.random.randint(1,window_size+1)
    start_idx = idx - target_window if (idx - target_window) > 0 else 0
    end_idx = idx + target_window
    targets_words = set(words[start_idx:idx]+words[idx+1:end_idx+1])
    return list(targets_words)


def get_batches(words, batch_size, window_size=5):  # batch 生成器
    batch_num = len(words) // batch_size  # 返回值形式为一个batch_size大小的句子中，所有的样本

    words = words[:batch_num * batch_size]
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_target(batch, i, window_size)
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = tf.placeholder(tf.int32,shape=[None],name='inputs')
    labels = tf.placeholder(tf.int32,shape=[None,None],name='labels')

vocab_size = len(int_to_vocab)
embedding_size = 64 # 嵌入维度

with train_graph.as_default():
    #嵌入权重矩阵
    embedding = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1,1))
    embed = tf.nn.embedding_lookup(embedding,inputs)

n_sampled = 100

with train_graph.as_default():
    weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    biases = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(weights, biases, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

# validation
with train_graph.as_default():
    valid_examples = [vocab_to_int['whileStatement'], vocab_to_int['forStatement']]
    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)  # 查找验证集的向量
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))  # 矩阵乘法

epochs = 10
batch_size = 1000
window_size = 5

with train_graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()

        for x, y in batches:
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                    "Iteration: {}".format(iteration),
                    "Avg. Training loss: {:.4f}".format(loss / 100),
                    "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "word2vec.ckpt")
    embed_mat = sess.run(embedding)

    wordvec_dict = {}
    for w, c in vocab_to_int.items():
        wordvec_dict[w] = embed_mat[c]
    print(wordvec_dict)

    output = open('word_vector.pkl', 'wb')
    pkl.dump(wordvec_dict, output)



# import tensorflow as tf
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess,"word2vec.ckpt")
#     graph = tf.get_default_graph()
#
