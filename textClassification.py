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
categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
print('total texts in train:',len(newsgroups_train.data))
print('total texts in test:',len(newsgroups_test.data))
print('text',newsgroups_train.data[0])
print('category:',newsgroups_train.target[0])
vocab = Counter()

for text in newsgroups_train.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
        
for text in newsgroups_test.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
print("Total words:",len(vocab))
total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index

word2index = get_word_2_index(vocab)

print("Index of the word 'the':",word2index['the'])
def text_to_vector(text):
    layer = np.zeros(total_words,dtype=float)
    for word in text.split(' '):
        layer[word2index[word.lower()]] += 1
        
    return layer

def category_to_vector(category):
    y = np.zeros((3),dtype=float)
    if category == 0:
        y[0] = 1.
    elif category == 1:
        y[1] = 1.
    else:
        y[2] = 1.
        
    return y
def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size:i*batch_size+batch_size]
    categories = df.target[i*batch_size:i*batch_size+batch_size]
    
    for text in texts:
        layer = text_to_vector(text) 
        batches.append(layer)
        
    for category in categories:
        y = category_to_vector(category)
        results.append(y)  
     
    return np.array(batches),np.array(results)
print("Each batch has 100 texts and each matrix has 119930 elements (words):",get_batch(newsgroups_train,1,100)[0].shape)
print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(newsgroups_train,1,100)[1].shape)
