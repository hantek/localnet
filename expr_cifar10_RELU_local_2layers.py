import os
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from dataset import CIFAR10
from classifier import LogisticRegression
from model import ReluAutoencoder
from preprocess import ZCA, SubtractMeanAndNormalizeH
from train import GraddescentMinibatch
from params import save_params, load_params, set_params, get_params

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
train_x = preprocess_function(train_x)
test_x = preprocess_function(test_x)

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
model = ReluAutoencoder(
    train_x.get_value().shape[1], hid_layer_sizes[0], 
    init_w = theano.shared(
        value=0.01 * train_x.get_value()[:hid_layer_sizes[0], :].T,
        name='w_reluae_0',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
)
"""
) + ReluAutoencoder(
    hid_layer_sizes[0], hid_layer_sizes[1],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[0] * hid_layer_sizes[1] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[0] * hid_layer_sizes[1])].reshape(
            hid_layer_sizes[0], hid_layer_sizes[1]
        ),
        name='w_reluae_1',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + ReluAutoencoder(
    hid_layer_sizes[1], hid_layer_sizes[2],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[1] * hid_layer_sizes[2] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[1] * hid_layer_sizes[2])].reshape(
            hid_layer_sizes[1], hid_layer_sizes[2]
        ),
        name='w_reluae_2',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + ReluAutoencoder(
    hid_layer_sizes[2], hid_layer_sizes[3],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[2] * hid_layer_sizes[3] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[2] * hid_layer_sizes[3])].reshape(
            hid_layer_sizes[2], hid_layer_sizes[3]
        ),
        name='w_reluae_3',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
"""
model.print_layer()
"""
# generate a fixed mask, layer 0
print "... generating l0 mask"
mask = numpy.zeros((train_x.get_value().shape[1], hid_layer_sizes[0]),
                   dtype=theano.config.floatX)
window_size = npy_rng.randint(layer_window_size[0][0], layer_window_size[0][1],
                              (hid_layer_sizes[0], ))
center_list_l0 = numpy.zeros((hid_layer_sizes[0], 3), dtype=theano.config.floatX)
for i, isize in enumerate(window_size):
    start_point = npy_rng.randint(0, 32 - isize, (2, ))
    stop_point = (start_point[0] + isize, start_point[1] + isize)
    imask = numpy.zeros((3, 32, 32))
    imask[:, start_point[0]:stop_point[0], start_point[1]:stop_point[1]] = 1
    mask[:, i] = imask.flatten()
    # compute for each hidden unit the center of local receptive filter
    center_list_l0[i] = (start_point[0] + isize * 1. / 2,
                         start_point[1] + isize * 1. / 2,
                         npy_rng.random_sample())

# range: [[0,23], [0,23], [0,94.51]], density: 1 neuron per volumn.
center_list_l0 -= 4
# 50000./24.**2
z_range = hid_layer_sizes[0] * 1.0 / center_list_l0[:, :2].max()**2
center_list_l0[:, 2] *= z_range
numpy.save(open("mask_l0.npy", 'wb'), mask)
"""
mask = numpy.load('mask_l0.npy')
mask_l0_theano = theano.shared(value=mask, name='mask_l0', borrow=True)
apply_mask_l0 = theano.function(
    inputs=[],
    updates={model.w : model.w * mask_l0_theano}
)
"""
# generate the mask for layer 1, according to center_list
# covering rate: 4*4*8000/(28*28) ~= 163,
# this number roughly equals filter number. 
print "... generating l1 mask"

mask = numpy.zeros((hid_layer_sizes[0], hid_layer_sizes[1]), dtype=theano.config.floatX)
window_size = npy_rng.randint(layer_window_size[1][0], layer_window_size[1][1],
                              (hid_layer_sizes[1], ))
center_list_l1 = numpy.zeros((hid_layer_sizes[1], 3), dtype=theano.config.floatX)

i = 0
for isize in window_size:
    start_point = npy_rng.randint(0, 24 - isize, (2, ))
    stop_point = (start_point[0] + isize, start_point[1] + isize)
    filter_position = npy_rng.random_sample() * z_range

    idx = 0
    for cx, cy, cz in center_list_l0:
        if (cx >= start_point[0] and cx <= stop_point[0]) \
            and (cy >= start_point[1] and cy <= stop_point[1]) \
            and (numpy.abs(cz - filter_position) <= 2):
            mask[idx, i] = 1
        idx += 1
    
    # compute for each hidden unit the center of local receptive filter
    center_list_l1[i] = (start_point[0] + isize * 1. / 2,
                         start_point[1] + isize * 1. / 2,
                         filter_position)
    i += 1

# range: [[0,12], [0,12], [0,138.888]], density: 1 neuron per volumn.
center_list_l1 -= 6
z_range = hid_layer_sizes[1] * 1.0 / center_list_l1[:, :2].max()**2
center_list_l0[:, 2] *= z_range / center_list_l1[:,2].max()
numpy.save(open("mask_l1.npy", 'wb'), mask)

mask = numpy.load('mask_l1.npy')
mask_l1_theano = theano.shared(value=mask, name='mask_l1', borrow=True)
apply_mask_l1 = theano.function(
    inputs=[],
    updates={model.models_stack[1].w : model.models_stack[1].w * mask_l1_theano}
)

# generate the mask for layer 2.
print "... generating l2 mask"

mask = numpy.zeros((hid_layer_sizes[1], hid_layer_sizes[2]), dtype=theano.config.floatX)
window_size = npy_rng.randint(layer_window_size[2][0], layer_window_size[2][1],
                              (hid_layer_sizes[2], ))
center_list_l2 = numpy.zeros((hid_layer_sizes[2], 3), dtype=theano.config.floatX)

i = 0
for isize in window_size:
    start_point = npy_rng.randint(0, 13 - isize, (2, ))
    stop_point = (start_point[0] + isize, start_point[1] + isize)
    filter_position = npy_rng.random_sample() * z_range
    
    idx = 0
    for cx, cy, cz in center_list_l1:
        if (cx >= start_point[0] and cx <= stop_point[0]) \
            and (cy >= start_point[1] and cy <= stop_point[1]) \
            and (numpy.abs(cz - filter_position) <= 2):
            mask[idx, i] = 1
        idx += 1
    
    # compute for each hidden unit the center of local receptive filter
    center_list_l2[i] = (start_point[0] + isize * 1. / 2,
                         start_point[1] + isize * 1. / 2,
                         filter_position)
    i += 1

numpy.save(open("mask_l2.npy", 'wb'), mask)

mask = numpy.load('mask_l2.npy')
mask_l2_theano = theano.shared(value=mask, name='mask_l2', borrow=True)
apply_mask_l2 = theano.function(
    inputs=[],
    updates={model.models_stack[2].w : model.models_stack[2].w * mask_l2_theano}
)

apply_mask = [apply_mask_l0, apply_mask_l1, apply_mask_l2]
# deeper layers are fully connected. 

# pdb.set_trace()
"""
print "Done."

#############
# PRE-TRAIN #
#############
"""
for i in range(len(hid_layer_sizes)):
    print "\n\nPre-training layer %d:" % i
    trainer = GraddescentMinibatch(
        varin=model.varin, data=train_x, 
        cost=model.models_stack[i].cost(),
        params=model.models_stack[i].params_private,
        supervised=False,
        batchsize=batchsize, learningrate=pretrain_lr, momentum=momentum,
        rng=npy_rng
    )

    init_lr = trainer.learningrate
    prev_cost = numpy.inf
    epc_cost = 0.
    patience = 0
    avg = 50
    crnt_avg = [numpy.inf, ] * avg
    hist_avg = [numpy.inf, ] * avg
    for step in xrange(pretrain_epc * 50000 / batchsize):
        # learn
        cost = trainer.step_fast(verbose_stride=500)
        if i < 3:
            apply_mask[i]()

        epc_cost += cost / (50000 / batchsize)
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

            epc_cost = 0.

    save_params(model, 'ZCARELUAE_50000_8000_1000_1600_1600_local_wsz6_12_4_8_3_6.npy')
load_params(model, 'ZCARELUAE_50000_8000_1000_1600_1600_local_wsz6_12_4_8_3_6.npy')
print "Done."
"""

#########################
# BUILD FINE-TUNE MODEL #
#########################

print "\n\n... building fine-tune model -- contraction 1"
model_ft = model + LogisticRegression(
    hid_layer_sizes[0], 10, npy_rng=npy_rng
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
"""
trainer = GraddescentMinibatch(
    varin=model_ft.varin, data=train_x, 
    truth=model_ft.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model_ft.models_stack[-1].cost() + \
         model_ft.models_stack[-1].weightdecay(weightdecay),
    params=model_ft.models_stack[-1].params, 
    batchsize=batchsize, learningrate=logreg_lr, momentum=momentum,
    rng=npy_rng
)

init_lr = trainer.learningrate
prev_cost = numpy.inf
for epoch in xrange(logreg_epc):
    cost = trainer.epoch()
    if cost >= prev_cost:
        trainer.set_learningrate(0.9 * trainer.learningrate)
        if trainer.learningrate < 1e-4 * init_lr:
            break
    prev_cost = cost
print "Done."
save_params(model, 'ZCARELUAE_50000_8000_1000_1600_10_local_wsz6_12_4_8_3_6.npy')
print "***error rate: train: %f, test: %f" % (
    train_error(), test_error()
)
"""
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
    apply_mask_l0()
    #apply_mask[1]()
    #apply_mask[2]()

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
