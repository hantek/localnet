import os
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from neurobricks.dataset import CIFAR10
from neurobricks.classifier import LogisticRegression
from neurobricks.layer import ReluConv2DLayer
from neurobricks.preprocess import ZCA, SubtractMeanAndNormalizeH
from neurobricks.train import GraddescentMinibatch
from neurobricks.params import save_params, load_params, set_params, get_params

import pdb


#######################
# SET SUPER PARAMETER #
#######################

zca_retain = 0.99
hid_layer_sizes = [25000, 15000, 2000, 1600]
batchsize = 100
layer_window_size = [(8, 12), (12, 16), (10, 12)]  # [(6, 10), (4, 8), (3, 6)]

momentum = 0.9
pretrain_lr = 1e-3
pretrain_epc = 400

logreg_lr = 0.5
logreg_epc = 1000
weightdecay = 0.01

finetune_lr = 5e-2
finetune_epc = 1000

print " "
print "zca energy retain =", zca_retain
print "hid_layer_sizes =", hid_layer_sizes
print "batchsize =", batchsize
print "layer_window_size =", layer_window_size
print "momentum =", momentum
print "pretrain             lr = %.2g, epc = %d" % (pretrain_lr, pretrain_epc)
print "logistic regression: lr = %f, wd = %f, epc = %d" % (logreg_lr, weightdecay, logreg_epc)
print "finetune:            lr = %f, epc = %d" % (finetune_lr, finetune_epc)

#############
# LOAD DATA #
#############

cifar10_data = CIFAR10()
train_x, train_y = cifar10_data.get_train_set()
test_x, test_y = cifar10_data.get_test_set()

print "\n... pre-processing"
preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
map_fun = theano.function([preprocess_model.varin], preprocess_model.output())

zca_obj = ZCA()
zca_obj.fit(map_fun(train_x), retain=zca_retain, whiten=True)
preprocess_model = preprocess_model + zca_obj.forward_layer
preprocess_function = theano.function([preprocess_model.varin], preprocess_model.output())
train_x = preprocess_function(train_x).reshape((50000, 32, 32, 3), order='F').swapaxes(1, 3)
test_x = preprocess_function(test_x).reshape((10000, 32, 32, 3), order='F').swapaxes(1, 3)


feature_num = train_x.shape[0] * train_x.shape[1]

train_x = theano.shared(value = train_x, name = 'train_x', borrow = True)
train_y = theano.shared(value = train_y, name = 'train_y', borrow = True)
test_x = theano.shared(value = test_x,   name = 'test_x',  borrow = True)
test_y = theano.shared(value = test_y,   name = 'test_y',  borrow = True)
print "Done."

#########################
# BUILD PRE-TRAIN MODEL #
#########################

print "... building pre-train model"
npy_rng = numpy.random.RandomState(123)
l0_n_in = (batchsize, 3, 32, 32)
l0_filter_shape=(40, l0_n_in[1], 8, 8)
model = ReluConv2DLayer(
    n_in=l0_n_in, filter_shape=l0_filter_shape, npy_rng=npy_rng
)
model.print_layer()
print "Done."

#########################
# BUILD FINE-TUNE MODEL #
#########################

print "\n\n... building fine-tune model -- contraction 1"
model_ft = model + LogisticRegression(
    None, 10, npy_rng=npy_rng
)
model_ft.print_layer()

# compile error rate counters:
index = T.lscalar()
truth = T.lvector('truth')
train_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model_ft.models_stack[-1].predict(), truth)),
    givens = {model_ft.varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_error():
    return numpy.mean([train_set_error_rate(i) for i in xrange(50000/batchsize)])

test_set_error_rate = theano.function(
    [index],
    T.mean(T.neq(model_ft.models_stack[-1].predict(), truth)),
    givens = {model_ft.varin : test_x[index * batchsize: (index+1) * batchsize],
              truth : test_y[index * batchsize: (index+1) * batchsize]},
)
def test_error():
    return numpy.mean([test_set_error_rate(i) for i in xrange(10000/batchsize)])
print "Done."

#############
# FINE-TUNE #
#############

print "\n\n... fine-tuning the whole network"
trainer = GraddescentMinibatch(
    varin=model_ft.varin, data=train_x, 
    truth=model_ft.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model_ft.models_stack[-1].cost() + \
         model_ft.models_stack[-1].weightdecay(weightdecay),
    params=model_ft.params, 
    batchsize=batchsize, learningrate=finetune_lr, momentum=momentum,
    rng=npy_rng
)

init_lr = trainer.learningrate
prev_cost = numpy.inf
epc_cost = 0.
patience = 0
avg = 50
crnt_avg = [numpy.inf, ] * avg
hist_avg = [numpy.inf, ] * avg
for step in xrange(finetune_epc * 50000 / batchsize):
    # learn
    cost = trainer.step_fast(verbose_stride=500)

    epc_cost += cost
    if step % (50000 / batchsize) == 0 and step > 0:
        # set stop rule
        ind = (step / (50000 / batchsize)) % avg
        hist_avg[ind] = crnt_avg[ind]
        crnt_avg[ind] = epc_cost
        if sum(hist_avg) < sum(crnt_avg):
            break
    
        # adjust learning rate
        if prev_cost <= epc_cost:
            patience += 1
        if patience > 10:
            trainer.set_learningrate(0.9 * trainer.learningrate)
            patience = 0
        prev_cost = epc_cost

        # evaluate
        print "***error rate: train: %f, test: %f" % (
            train_error(), test_error())
        
        epc_cost = 0.
print "Done."
print "***FINAL error rate, train: %f, test: %f" % (
    train_error(), test_error()
)
save_params(model, __file__.split('.')[0] + '_params.npy')

pdb.set_trace()
