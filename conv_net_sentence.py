# *_*coding:utf-8 *_*
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
warnings.filterwarnings("ignore")

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def train_conv_net(datasets,
                   U,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   use_valid_set=True,
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
    parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden_units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
        , ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch)]
    print (parameters)

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value=U, name="Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w),
                                        filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, 1)
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations,
                            dropout_rates=dropout_rate)
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        params += [Words]
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])
        extra_data = train_set[:extra_data_num]
        new_data = np.append(datasets[0], extra_data, axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = int(np.round(n_batches * 0.9))
    if len(datasets) == 3:
        use_valid_set = True
        train_set = new_data
        val_set = datasets[1]
        train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))
        val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
        test_set_x = datasets[2][:, :img_h]
        test_set_y = np.asarray(datasets[2][:, -1], "int32")
        n_val_batches = int(val_set.shape[0] / batch_size)
        val_model = theano.function([index], classifier.errors(y),
                                    givens={
                                        x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                        y: val_set_y[index * batch_size: (index + 1) * batch_size]})
    else:
        test_set_x = datasets[1][:, :img_h]
        test_set_y = np.asarray(datasets[1][:, -1], "int32")
        if use_valid_set:
            train_set = new_data[:n_train_batches * batch_size, :]
            val_set = new_data[n_train_batches * batch_size:, :]
            train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))

            val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
            n_val_batches = n_batches - n_train_batches
            val_model = theano.function([index], classifier.errors(y),
                                        givens={
                                            x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                            y: val_set_y[index * batch_size: (index + 1) * batch_size]})
        else:
            train_set = new_data[:, :]
            train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))
        test_model = theano.function([index], classifier.errors(y),
                                     givens={
                                         x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: train_set_y[index * batch_size: (index + 1) * batch_size]})
        train_model = theano.function([index], cost, updates=grad_updates,
                                      givens={
                                          x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                          y: train_set_y[index * batch_size:(index + 1) * batch_size]})
        test_pred_layers = []
        test_size = test_set_x.shape[0]
        test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Words.shape[1]))
        for conv_layer in conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = classifier.predict(test_layer1_input)
        test_error = T.mean(T.neq(test_y_pred, y))
        test_model_all = theano.function([x, y], test_error)

        epoch = 0
        best_val_perf = 0
        val_perf = 0
        test_perf = 0
        cost_epoch = 0
        while (epoch < n_epochs):
            epoch = epoch + 1
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)
                    set_zero(zero_vec)
            train_losses = [test_model(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)
            print('epoch %i, train perf %f %%, val perf %f' % (epoch, train_perf * 100., val_perf * 100.))
            if val_perf >= best_val_perf:
                test_loss = test_model_all(test_set_x, test_set_y)
                test_perf = 1 - test_loss
        return test_perf

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
        
        updates = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        gparams = []
        for param in params:
            empty = np.zeros_like(param.get_value())
            exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
            gp = T.grad(cost, param)
            exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
            gparams.append(gp)
        for param, gp in zip(params, gparams):
            exp_sg = exp_sqr_grads[param]
            exp_su = exp_sqr_ups[param]
            up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
            updates[exp_sg] = up_exp_sg
            step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
            updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
            stepped_param = param + step
            if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param
        return updates