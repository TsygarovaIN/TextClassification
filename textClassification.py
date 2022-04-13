import tensorflow as tf

my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    x = tf.constant([1,3,6]) 
    y = tf.constant([1,1,1])
    op = tf.add(x,y)
    result = sess.run(fetches=op)
    print(result)
    
vocab = Counter()

text = "Hi from Brazil"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1
        
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word] = i
        
    return word2index
    
word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words),dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1
    
print("Hi from Brazil:", matrix)

matrix = np.zeros((total_words),dtype=float)
text = "Hi"
for word in text.split():
    matrix[word2index[word.lower()]] += 1
    
print("Hi:", matrix)
