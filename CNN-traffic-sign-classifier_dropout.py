# carnd-traffic-sign-classifier-project/Traffic_Sign_Classifier-Copy1.ipynb
# Load pickled data
import pickle
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# Fill this in based on where you saved the training and testing data

# training_file = 'C:/Users/chaoqun.shan/carnd-traffic-sign-classifier-project/traffic-signs-data/train.p'
# validation_file = 'C:/Users/chaoqun.shan/carnd-traffic-sign-classifier-project/traffic-signs-data/valid.p'
# testing_file = 'C:/Users/chaoqun.shan/carnd-traffic-sign-classifier-project/traffic-signs-data/test.p'
label_file = 'E:/Udacity_Autonomous_Driving/Term1/traffic-signs-data/signnames.csv'
training_file = 'E:/Udacity_Autonomous_Driving/Term1/traffic-signs-data/train.p'
validation_file = 'E:/Udacity_Autonomous_Driving/Term1/traffic-signs-data/valid.p'
testing_file = 'E:/Udacity_Autonomous_Driving/Term1/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train_pure, y_train = train['features'], train['labels']
X_valid_pure, y_valid = valid['features'], valid['labels']
X_test_pure, y_test = test['features'], test['labels']

'''
Step 1: Dataset Summary & Exploration
'''
### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results
assert (len(X_train_pure) == len(y_train))
assert (len(X_valid_pure) == len(y_valid))
assert (len(X_test_pure) == len(y_test))

# Number of training examples
n_train = len(X_train_pure)

# Number of validation examples
n_validation = len(X_valid_pure)

# Number of testing examples.
n_test = len(X_test_pure)

# What's the shape of an traffic sign image?
image_shape = X_train_pure[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

'''Visualize Data'''

# %matplotlib inline

# index = random.randint(0, len(X_train))
# image = X_train[index].squeeze()

# plt.figure(figsize=(1, 1))
# plt.imshow(image, cmap="gray")
# print(y_train[index])
def NormImage(ImgSet):
    Num_ImgSet = len(ImgSet)
    NewSet = []
    if Num_ImgSet > 1:
        pixel128 = np.ones_like(ImgSet[0]) * 128

        for i in range(Num_ImgSet):
            NewImg = (pixel128 - ImgSet[i]) / 128
            NewSet.append(NewImg)
    else:
        pixel128 = np.ones_like(ImgSet) * 128
        NewSet = (pixel128 - ImgSet) / 128
    return NewSet

def Gray(ImgSet):
    Num_ImgSet = len(ImgSet)
    shape = ImgSet[0].shape
    NewSet = []
    if Num_ImgSet > 1:
        # pixel128 = np.ones_like(ImgSet[0]) * 128
        NewImg = np.zeros((shape[0], shape[1], 1))

        for i in range(Num_ImgSet):
            img = ImgSet[i]
            tmp = np.array([np.dot(img[..., :3], [0.299, 0.587, 0.114])])
            NewImg = np.rollaxis(tmp, 0, 3) #(matrix, 需调整的轴, 目标位置)
            NewSet.append(NewImg)

    else:
        print('Input should be dataset instead of a single image')
    return NewSet
X_train = Gray(X_train_pure)
X_valid = Gray(X_valid_pure)
X_test = Gray(X_test_pure)
'''Step 2: Design and Test a Model Architecture'''
# Pad images with 0s
# X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# X_valid = np.pad(X_valid, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
# X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

X_train, y_train = shuffle(X_train, y_train)

'''Model Architecture'''

EPOCHS = 20
BATCH_SIZE = 128

def Sign(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    ConvStrides = [1, 1]
    PoolStrides = [2, 2]
    L1Filter = [5, 5, 1] # [Filter height, Filter width, Input depth]
    L1Output = [(32-L1Filter[0]+1) / ConvStrides[0], (32-L1Filter[1]+1) / ConvStrides[1], 6 ]  # VALID Padding output computation formula
    L2Filter = [3, 3, L1Output[2]]
    L2Output = [(L1Output[0]/PoolStrides[0] - L2Filter[0] + 1)/ConvStrides[0],
                (L1Output[1]/PoolStrides[1] - L2Filter[1] + 1)/ConvStrides[1], 20]
    L3Filter = [3, 3, L2Output[2]]
    L3Output = [(L2Output[0]/PoolStrides[0] - L3Filter[0] + 1)/ConvStrides[0],
                (L2Output[1]/PoolStrides[1] - L3Filter[1] + 1)/ConvStrides[1], 60]
    L4Input = int((L3Output[0]/PoolStrides[0]) * (L3Output[1]/PoolStrides[1]) * L3Output[2])
    L4Output = 160
    L5Output = 80


    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(L1Filter[0], L1Filter[1], L1Filter[2], L1Output[2]), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(L1Output[2]))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x16x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x16x6 Output = 12x12x20.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(L2Filter[0], L2Filter[1], L2Filter[2], L2Output[2]), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(L2Output[2]))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 12x12x20. Output = 6x6x20.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Layer 3: Convolutional. Input = 6x6x20. Output = 4x4x60.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(L3Filter[0], L3Filter[1], L3Filter[2], L3Output[2]), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(L3Output[2]))
    conv3 = tf.nn.conv2d(conv2, conv3_W, strides=[1, ConvStrides[0], ConvStrides[1], 1], padding='VALID') + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Pooling. Input = 4x4x60. Output = 2x2x60.
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, PoolStrides[0], PoolStrides[1], 1], padding='VALID')

    # Flatten. Input = 2x2x60. Output = 240.
    fc0 = flatten(conv3)

    # Layer 4: Fully Connected. Input = 240. Output = 160.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(L4Input, L4Output), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(L4Output))
    fc0 = tf.nn.dropout(fc0, keep_prob)
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 5: Fully Connected. Input = 160. Output = 80.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(L4Output, L5Output), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(L5Output))
    # fc1 = tf.nn.dropout(fc1, keep_prob)
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 6: Fully Connected. Input = 80. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(L5Output, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    # fc2 = tf.nn.dropout(fc2, keep_prob)
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)

'''Training Pipeline'''
rate = 0.001

logits = Sign(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

'''Model Evaluation'''
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


'''Train the Model'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    AccuracySet = []
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        AccuracySet.append(validation_accuracy * 100)


    saver.save(sess, './Traffic-Sign-CNNmodel')
    print("Model saved")

'''Plot Accurancy'''
xData = [i for i in range(EPOCHS)]
xlabel = [i for i in range(1,EPOCHS,2)]
xlableSet = []
for i in range(len(xlabel)):
    xl = '{}'.format(xlabel[i])
    xlableSet.append(xl)
plt.figure()
plt.plot(xData, AccuracySet)
plt.xticks([i for i in range(0,EPOCHS,2)], tuple(xlableSet))
plt.xlabel('Echo')
plt.ylabel('Accurancy %')
plt.ylim(65, 95)
plt.grid()
plt.show()

# '''Load and Output the Images'''
num_img = 10
test_imgs = X_test_pure[:num_img]
test_labels = y_test[:num_img]
plt.figure(figsize=(20, 50))

for idx in range(num_img):
    plt.subplot(2, num_img, idx + 1)
    plt.imshow(test_imgs[idx], cmap="gray")
    plt.title("Label={}".format(test_labels[idx]))

plt.show()
print("Test image shape = {}".format(test_imgs[0].shape))

# '''Predict the Sign Type and analyze performance'''
test_imgs = X_test[:num_img]

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(test_imgs, test_labels)
#     test_accuracy = sess.run(accuracy_operation, feed_dict={x: test_imgs, y: test_labels})

    print("Test Accuracy = {:.3f}".format(test_accuracy))

