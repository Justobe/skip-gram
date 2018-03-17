import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
pkl_file = open('word_vector.pkl','rb')
word_vectors = pickle.load(pkl_file)
word_vectors.pop('')

words_list = list(word_vectors.keys())
vector_list = list(word_vectors.values())
x_y = TSNE(n_components=2).fit_transform(vector_list)
i = 0
for x,y in x_y:
    plt.scatter(x, y)
    plt.annotate(words_list[i], xy=(x, y),)
    i = i + 1
plt.show()

