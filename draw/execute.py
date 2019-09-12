# import libraries
import tensorflow as tf
from tensorflow.examples.tutorials import mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from PIL import Image
import os

# hack for duplicated library on MacOS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# reset tf package
tf.pack = tf.stack
tf.select = tf.where
tf.batch_matmul = tf.matmul

# define fully-connected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias

# merge image
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx / size[1])
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

# save image
def save(filename, image_array):
    Image.fromarray((image_array*255).astype('uint8'), mode='L').convert('RGB').save(filename)

# DRAW implementation
class draw_model():
    def __init__(self):
        # 1. download MNIST dataset to our folder
        self.mnist = input_data.read_data_sets("data/", one_hot=True)
        self.n_samples = self.mnist.train.num_examples

        # 2. set model parameters
        # image width and height
        self.image_size = 28
        # read glimpse grid width/height
        self.attention_n = 5
        # number of hidden units / output size in LSTM
        self.n_hidden = 256
        # QSampler output size
        self.n_output = 10
        # MNIST generation sequence length
        self.sequence_length = 10
        # training minibatch size
        self.batch_size = 64
        # workaround for variable_scope(reuse=True)
        self.share_parameters = False

        # 3. build model
        # input(batch_size * image_size)
        self.images = tf.placeholder(tf.float32, [None, 784])
        # Qsample noise
        self.noise = tf.random_normal((self.batch_size, self.n_output), mean=0, stddev=1)
        # encoder 0p
        self.lstm_enc = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)
        # decoder 0p
        self.lstm_dec = tf.nn.rnn_cell.LSTMCell(self.n_hidden, state_is_tuple=True)

        # 4. define state variables
        # sequence of canvases
        self.seq_canvas = [0] * self.sequence_length
        self.mu, self.logsigma, self.sigma = [0] * self.sequence_length, [0] * self.sequence_length, [0] * self.sequence_length

        # 5. initalize states
        h_dec_prev = tf.zeros((self.batch_size, self.n_hidden))
        enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        dec_state = self.lstm_dec.zero_state(self.batch_size, tf.float32)

        # 6. construct the unrolled computational graph
        x = self.images
        for t in range(self.sequence_length):
            # error image + original image
            c_prev = tf.zeros((self.batch_size, self.image_size**2)) if t == 0 else self.seq_canvas[t-1]
            x_hat = x - tf.sigmoid(c_prev)
            # read the image
            r = self.read_attention(x, x_hat, h_dec_prev)
            #  sanity check
            print(r.get_shape())
            # encode to guassian distribution
            self.mu[t], self.logsigma[t], self.sigma[t], enc_state = self.encode(enc_state, tf.concat([r, h_dec_prev], 1))
            # sample from the distribution to get z
            z = self.sampleQ(self.mu[t], self.sigma[t])
            # sanity check
            print(z.get_shape())
            # retrieve the hidden layer of RNN
            h_dec, dec_state = self.decode_layer(dec_state, z)
            # sanity check
            print(h_dec.get_shape())
            # map from hidden layer
            self.seq_canvas[t] = c_prev + self.write_attention(h_dec)
            h_dec_prev = h_dec
            # from now no, share variables
            self.share_parameters = True

        # 7. loss function
        self.generated_images = tf.nn.sigmoid(self.seq_canvas[-1])
        self.generation_loss = tf.reduce_mean(-tf.reduce_sum(self.images * tf.log(1e-10 + self.generated_images) + (1 - self.images) * tf.log(1e-10 + 1 - self.generated_images), 1))

        kl_terms = [0] * self.sequence_length
        for t in range(self.sequence_length):
            mu2 = tf.square(self.mu[t])
            sigma2 = tf.square(self.sigma[t])
            logsigma = self.logsigma[t] ###########
            # each kl term is (1 * minibatch)
            kl_terms[t] = 0.5 * tf.reduce_sum(mu2 + sigma2 - 2 * logsigma, 1) - self.sequence_length * 0.5
        self.latent_loss = tf.reduce_mean(tf.add_n(kl_terms))
        self.cost = self.generation_loss + self.latent_loss

        # 8. optimization
        optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        grads = optimizer.compute_gradients(self.cost)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)
        self.train_op = optimizer.apply_gradients(grads)

        # 9. intialize session
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    #training function
    def train(self):
        for i in range(20000):
            xtrain, _  = self.mnist.train.next_batch(self.batch_size)
            seq_canvas, gen_loss, lat_loss, _ = self.sess.run([self.seq_canvas, self.generation_loss, self.latent_loss, self.train_op], feed_dict={self.images: xtrain})
            print("iter: %d genloss: %f latloss: %f" % (i, gen_loss, lat_loss))
            if i % 500 == 0:
                seq_canvas = 1.0 / (1.0 + np.exp(-np.array(seq_canvas)))
                for sc_iter in range(10):
                    results = seq_canvas[sc_iter]
                    results_square = np.reshape(results, [-1, 28, 28])
                    print(results_square.shape)
                    save("result/" + str(i) + "-step--" + str(sc_iter) + ".jpg", merge(results_square, [8,8]))

    # Eric Jang's main functions
    # ------------------------------
    # locate where to put attention filters on hidden layers
    def attn_window(self, scope, h_dec):
        with tf.variable_scope(scope, reuse=self.share_parameters):
            parameters = dense(h_dec, self.n_hidden, 5)
        # center of 2d gaussian on a scale of -1 to 1
        gx_, gy_, log_sigma2, log_delta, log_gamma = tf.split(parameters, np.int(5), np.int(1))
        # move gx/gy to be a scale of -imagesize to +imagesize
        gx = (self.image_size + 1) / 2 * (gx_ + 1)
        gy = (self.image_size + 1) / 2 * (gy_ + 1)

        sigma2 = tf.exp(log_sigma2)
        # distance between patches
        delta = (self.image_size - 1) / ((self.attention_n - 1) * tf.exp(log_delta))
        # returns [Fx, Fy, gamma]
        return self.filterbank(gx, gy, sigma2, delta) + (tf.exp(log_gamma),)

    # construct patches of gaussian filters
    def filterbank(self, gx, gy, sigma2, delta):
        # 1 * N, look like [[0, 1, 2, 3, 4]]
        grid_i = tf.reshape(tf.cast(tf.range(self.attention_n), tf.float32), [1, -1])
        # individual patches centers
        mu_x = gx + (grid_i - self.attention_n / 2 - 0.5) * delta
        mu_y = gy + (grid_i - self.attention_n / 2 - 0.5) * delta
        mu_x = tf.reshape(mu_x, [-1, self.attention_n, 1])
        mu_y = tf.reshape(mu_y, [-1, self.attention_n, 1])
        # 1 * 1 * imagesize, looks like [[[0, 1, 2, 3, 4, ..., 27]]]
        im = tf.reshape(tf.cast(tf.range(self.image_size), tf.float32), [1, 1, -1])
        # list of gaussian curves for x and y
        sigma2 = tf.reshape(sigma2, [-1, 1, 1])
        Fx = tf.exp(-tf.square((im - mu_x) / (2 * sigma2)))
        Fy = tf.exp(-tf.square((im - mu_x) / (2 * sigma2)))
        # normalize area-under-curve
        Fx = Fx / tf.maximum(tf.reduce_sum(Fx, 2, keep_dims=True), 1e-8)
        Fy = Fy / tf.maximum(tf.reduce_sum(Fy, 2, keep_dims=True), 1e-8)
        return Fx, Fy

    # read operation without attention
    def read_basic(self, x, x_hat, h_dec_prev):
        return tf.concat([x_hat, x], 1)

    # read operation with attention
    def read_attention(self, x, x_hat, h_dec_prev):
        Fx, Fy, gamma = self.attn_window("read", h_dec_prev)
        # apply parameters for patch of gaussian filters
        def filter_img(img, Fx, Fy, gamma):
            Fxt = tf.transpose(Fx, perm=[0, 2, 1])
            img = tf.reshape(img, [-1, self.image_size, self.image_size])
            # apply the gaussian patches
            glimpse = tf.batch_matmul(Fy, tf.batch_matmul(img, Fxt))
            glimpse = tf.reshape(glimpse, [-1, self.attention_n**2])
            # scale using the gamma parameter
            return glimpse * tf.reshape(gamma, [-1, 1])

        x = filter_img(x, Fx, Fy, gamma)
        x_hat = filter_img(x_hat, Fx, Fy, gamma)
        return tf.concat([x, x_hat], 1)

    # encoder function for attention patch
    def encode(self, prev_state, image):
        # update the RNN with our image
        with tf.variable_scope("encoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_enc(image, prev_state)

        # map the RNN hidden state to latent variables
        with tf.variable_scope("mu", reuse=self.share_parameters):
            mu = dense(hidden_layer, self.n_hidden, self.n_output)
        with tf.variable_scope("sigma", reuse=self.share_parameters):
            logsigma = dense(hidden_layer, self.n_hidden, self.n_output)
            sigma = tf.exp(logsigma)
        return mu, logsigma, sigma, next_state

    # 
    def sampleQ(self, mu, sigma):
        return mu + sigma*self.noise

    # decoder function
    def decode_layer(self, prev_state, latent):
        # update decoder RNN using our latent variable
        with tf.variable_scope("decoder", reuse=self.share_parameters):
            hidden_layer, next_state = self.lstm_dec(latent, prev_state)
        return hidden_layer, next_state
    
    # write operation without attention
    def write_basic(self, hidden_layer):
        # map RNN hidden state to image
        with tf.variable_scope("write", reuse=self.share_parameters):
            decoded_image_portion = dense(hidden_layer, self.n_hidden, self.image_size**2)
        return decoded_image_portion

    # write operation with attention
    def write_attention(self, hidden_layer):
        with tf.variable_scope("writefW", reuse=self.share_parameters):
            w = dense(hidden_layer, self.n_hidden, self.attention_n**2)

        w = tf.reshape(w, [self.batch_size, self.attention_n, self.attention_n])
        Fx, Fy, gamma = self.attn_window("write", hidden_layer)
        Fyt = tf.transpose(Fy, perm=[0, 2, 1])
        wr = tf.batch_matmul(Fyt, tf.batch_matmul(w, Fx))
        wr = tf.reshape(wr, [self.batch_size, self.image_size**2])
        return wr * tf.reshape(1.0/gamma, [-1, 1])
    
model = draw_model()
model.train()
