"""Compute forward and backward propagation for layers.
"""
import math
import numpy as np
import im2col as im2


def fc_fw(inputs, weight, bias):
    """Compute forward propagation for a fully-connected layer.

    Args:
        inputs: The input image stack of the layer, of the shape (width, height,
            depth).
        weight: A numpy array of shape (n_in, n_out), where n_in = width *
            height * depth.
        bias: A numpy array of shape (n_out,).

    Returns:
        outputs: The output of the layer, of the shape (n_out,).
        cache: The tuple (inputs, weight, bias).
    """
    input_vec = np.reshape(inputs, -1)
    outputs = np.dot(weight.T, input_vec) + bias
    cache = (inputs, weight, bias)
    return outputs, cache


def fc_bk(dout, cache):
    """Compute backward propagation for a fully-connected layer.

    Args:
        dout: The gradient w.r.t. the output of the layer, of shape (n_out,).
        cache: The cache in the return of fc_forward.

    Returns:
        din: The gradient w.r.t. the input of the layer, of shape (width,
            height, depth).
        dweight: The gradient w.r.t. the weight, of shape (n_in, n_out).
        dbias: The gradient w.r.t. the bias, of shape (n_out,).
    """
    inputs, weight, _ = cache
    din_vec = np.dot(weight, dout)
    din = np.reshape(din_vec, inputs.shape)
    dweight = np.outer(inputs, dout)
    dbias = dout
    return din, dweight, dbias


def softmax(inputs, label):
    """Compute softmax function, loss function and its gradient.

    Args:
        inputs: The input scores, of shape (c,), where x[j] is the score for the
            j-th class.
        label: The true label of the input, taking from {0, 1, ..., c-1}.

    Returns:
        probs: The probabilities for each class.
        loss: The value of loss function.
        din: The gradient of loss function.
    """
    input_shift = inputs - max(inputs)
    sum_exp = sum(np.exp(input_shift))
    log_probs = input_shift - math.log(sum_exp)
    probs = np.exp(log_probs)
    loss = -log_probs[label]
    din = probs.copy()
    din[label] -= 1
    return probs, loss, din


def relu_fw(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache


def relu_bk(X, cache):
    dX = np.zeros(X.shape)
    dX[X > 0] = 1
    return dX


def conv_fw(inputs, weight, bias):
    n_filters, d_filter, h_filter, w_filter = weight.shape
    d_x, h_x, w_x = inputs.shape
    h_out = h_x - h_filter + 1
    w_out = w_x - w_filter + 1

    h_out, w_out = int(h_out), int(w_out)

    inputs_col = im2.im2col(inputs, h_filter, w_filter)
    weights_col = weight.reshape(n_filters, -1)

    out = np.dot(weights_col, inputs_col) + bias
    out = out.reshape(n_filters, h_out, w_out)
    out = out.transpose(0, 1, 2)

    cache = (inputs, weight, bias, inputs_col)

    return out, cache
