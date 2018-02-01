import tensorflow as tf
import os


def sliced_argmax(values, slices, k=1):
    """
    Computes top-k of values in each slice.
    :param values: matrix of shape [m,n]
    :param slices: vector of shape [m] containing start indices for each slice.
    :param k: take this many elements with largest values from each slice
    :returns: batch_scores,batch_indices:
        - batch_scores[m,k] - top-beam_size values from logP corresponding to
        - batch_indices[m,k] - indices of batch_scores in each respective slice (first value in each slice has index 0!)

    For any slice contains less than k elements, batch_scores would be padded with -inf, batch_indices - with -1
    If values.shape[1] != 1, batch_indices will still be 1-dimensional, satisfying the following property:
        - batch_scores,batch_indices = sliced_argmax(values,slices,k)
        - start, end = slices[i], slices[i+1]
        - tf.equals(batch_scores == tf.reshape(values[start:end,:],[-1])[batch_indices])  #this is True for all indices
    """
    assert values.shape.ndims in (None, 2), "values must be 2-dimensional"
    assert slices.shape.ndims in (None, 1), "slices must be 1-dimensional"

    def gpu_argmax(values, slices, k):
        return tf.cond(tf.equal(k, 1),
                       lambda: sliced_argmax1(values, slices),
                       lambda: tuple(v[:, :k] for v in sliced_argmax16(values, slices)))

    def cpu_argmax(values, slices, k):
        return sliced_argmax_cpu(values, slices, k)

    return tf.cond(tf.greater(k, 16),
                   lambda: cpu_argmax(values, slices, k),
                   lambda: gpu_argmax(values, slices, k))


@tf.RegisterGradient("SlicedArgmax")
@tf.RegisterGradient("SlicedArgmax1")
@tf.RegisterGradient("SlicedArgmax16")
def _sliced_argmax_gradient(op, grad_best_scores, grad_best_ix):
    scores, slices = op.inputs[:2]  # ignore k for SlicedArgmax
    best_scores, best_ix = op.outputs
    best_ix.set_shape([None, None])

    n_items = tf.shape(scores)[1]
    row_ix = tf.floordiv(best_ix, n_items) + slices[:, None]
    col_ix = tf.mod(best_ix, n_items)
    selector_ix = tf.stack([tf.reshape(row_ix, [-1]),
                            tf.reshape(col_ix, [-1])], axis=1)  # a vector of pairs [row_i, col_i]

    mask = tf.not_equal(best_ix, -1)
    selected_grads = tf.boolean_mask(grad_best_scores, mask)
    selector_ix = tf.boolean_mask(selector_ix, tf.reshape(mask, [-1]))

    grad_scores = tf.scatter_nd(selector_ix, selected_grads, shape=tf.shape(scores))
    if len(op.inputs) == 2:  # SlicedArgmax1, SlicedArgmax16
        return grad_scores, tf.zeros_like(slices)
    else:  # SlicedArgmax
        return grad_scores, tf.zeros_like(slices), tf.zeros_like(op.inputs[2])


def _have_gpu():
    if tf.VERSION >= '1.4':
        device_list = tf.get_default_session().list_devices()
    else:
        device_list = list_devices(tf.get_default_session())
    return len([x for x in device_list if x.device_type == 'GPU']) != 0


def _postprocess(obj, k=None):
    best_scores, best_ix = obj.out_val,  obj.out_idx
    shape_1 = k if isinstance(k, int) else None
    best_scores.set_shape([None, shape_1])
    best_ix.set_shape([None, shape_1])
    return (best_scores, best_ix)


def sliced_argmax_cpu(logP, slices, k):
    result = get_library().sliced_argmax(logP, slices, k)
    return _postprocess(result, k=k)


def sliced_argmax1(logP, slices):
    result = get_library().sliced_argmax1(logP, slices)
    return _postprocess(result, k=1)


def sliced_argmax16(logP, slices):
    result = get_library().sliced_argmax16(logP, slices)
    return _postprocess(result, k=16)


# list_devices is broken in TF 1.3
# remove after move on TF >= 1.4
# https://github.com/tensorflow/tensorflow/issues/13359
from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.framework import device
from tensorflow.python.framework import errors
from tensorflow.python.client.session import _DeviceAttributes

def list_devices(session):
    with errors.raise_exception_on_not_ok_status() as status:
        if session._created_with_new_api:
            raw_device_list = tf_session.TF_SessionListDevices(
                session._session, status)
        else:
            raw_device_list = tf_session.TF_DeprecatedSessionListDevices(
                session._session, status)
        device_list = []
        size = tf_session.TF_DeviceListCount(raw_device_list)
        for i in range(size):
            name = tf_session.TF_DeviceListName(raw_device_list, i, status)
            device_type = tf_session.TF_DeviceListType(raw_device_list, i, status)
            memory = 0   # tf_session.TF_DeviceListMemoryBytes(raw_device_list, i, status)
            device_list.append(_DeviceAttributes(name, device_type, memory))
        tf_session.TF_DeleteDeviceList(raw_device_list)
        return device_list


_library = None

def get_library():
    global _library
    if _library is None:
        so_fname = 'sliced_argmax.so' if _have_gpu() else 'sliced_argmax_cpu.so'
        so_path = os.path.join(*(os.path.split(__file__)[:-1] + (so_fname,)))
        _library = tf.load_op_library(so_path)

    return _library
