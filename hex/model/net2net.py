"""
Python file to define the functions that perform the net2net
knowledge transfer operations along with the functions that
define the flow from the shallow to the medium sized model and the
medium sized model to the deep model.

:references: https://keras.io/examples/mnist_net2net/
"""

from tensorflow.keras import backend as K
import numpy as np


def identity_batchnorm(model_input_tensor, batch_norm_input_tensor, data, axes=(0, 1, 2), epsilon=0.001):
    """
    Perform forward propagation on the provided data to infer the population
    statistics at the input tensor of a newly added BatchNormalization layer.

    Return weights for the BatchNormalization layer that denormalizes the normalization
    done by BatchNormalization, such that the newly added layer maps an identity function.

    When multiple BatchNormalization layers are added simultaneously, this function must
    be called in the order from input to output. This prevents scaling issues in later
    layers and ensures that all following BatchNorm layers can infer the correct statistics.

    :param model_input_tensor: Input tensor of the Keras Model.
    :param batch_norm_input_tensor: Input tensor of the newly added BatchNormalization layer
                                    that is present within the model.
    :param data: np.array array of board states to infer the BatchNorm's statistics on.
    :param axes: tuple The axes to perform the normalization over (0, 1, 2) is default for Conv2D layers.
    :param epsilon: float The epsilon value used by the BatchNorm layer.
    :see: 'Net2Net: Accelerating Learning via Knowledge Transfer'
           by Tianqi Chen, Ian Goodfellow, and Jonathon Shlens
    """
    f_feedforward = K.function(model_input_tensor, batch_norm_input_tensor)
    batch_size = 64

    means = list()
    vars = list()
    for batch in range(0, len(data), batch_size):
        results = f_feedforward(data[batch:batch+batch_size])
        means.append(np.mean(results, axis=axes))
        vars.append(np.var(results, axis=axes, ddof=1))

    pop_mean = np.mean(means, axis=0)
    sample_var = np.mean(vars, axis=0)

    # gamma, beta, mu, sigma
    return [
        np.sqrt(sample_var + epsilon),  # Undo the scaling by the population variance denominator.
        pop_mean,  # Undo the translation by the population mean.
        pop_mean,
        sample_var
    ]


def identity_fc(weights):
    """
    Get initial Dense weights for deepening a trained keras model such
    that the new weight matrix maps an identity filter.

    :param weights: weight matrix of Keras Dense layer preceding the new layer.
    :see: 'Net2Net: Accelerating Learning via Knowledge Transfer'
           by Tianqi Chen, Ian Goodfellow, and Jonathon Shlens
    """
    return np.eye(weights.shape[1])


def identity_convolution(kernel):
    """
    Get initial conv2d weights for deepening a trained
    keras model such that the kernel maps an identity filter.

    :param teacher_w: kernel of Keras conv2d layer preceding the new layer.
    :see: https://keras.io/examples/mnist_net2net/
    """
    filter_height, filter_width, input_dim, channels = kernel.shape
    conv_weights = np.zeros(kernel.shape)

    for i in range(channels):
        conv_weights[(filter_height - 1) // 2, (filter_width - 1) // 2, i, i] = 1.

    conv_bias = np.zeros(channels)

    return [conv_weights, conv_bias]


def wider2net_conv2d(teacher_w1, teacher_b1, teacher_w2, new_width, index=None, add_noise=True):
    """
    Get initial weights for widening a trained keras conv2d layer
    such that f(x;w) = f(x;w') forall x, where w is the current set of
    weights and w' is the set of weights with the widened layer.

    Ensure that new_width > teacher_w1.channels.

    :param teacher_w1: kernel of Keras conv2d layer to become wider.
    :param teacher_b1: bias of conv2d layer to become wider.
    :param teacher_w2: kernel of the following conv2d layer.
    :param new_width: new filters for the wider conv2d layer.
    :param index: Prespecified random mapping for widening.
    :param add_noise: Boolean whether to add noise to break symmetry.

    :see: https://keras.io/examples/mnist_net2net/
    """
    assert teacher_w1.shape[0] == teacher_w2.shape[1], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[3] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[3], (
        'new width (filters) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[3]

    if index is None:
        index = np.random.randint(teacher_w1.shape[3], size=n)
    factors = np.bincount(index)[index] + 1.
    new_w1 = teacher_w1[..., index]
    new_b1 = teacher_b1[index]
    new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)

    # Add small noise to break symmetry, so that student model will have full capacity later
    noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape) if add_noise else 0
    student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=2)
    student_w2[:, :, index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def wider2net_fc(teacher_w1, teacher_b1, teacher_w2, new_width, index=None, add_noise=True):
    """
    Get initial weights for a wider fully connected (dense) layer
    with a bigger nout, by 'random-padding' or 'net2wider'.

    :param teacher_w1: Weight matrix of Dense layer to become wider,
          of shape (nin1, nout1)
    :param teacher_b1: Bias of Dense layer to become wider.
    :param teacher_w2: Weight matrix of the following Dense layer
    :param new_width: New number of output nodes > number old output nodes.
    :param index: Prespecified random mapping for widening.
    :param add_noise: Boolean whether to add noise to break symmetry.

    :see: https://keras.io/examples/mnist_net2net/
    """
    assert teacher_w1.shape[1] == teacher_w2.shape[0], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[1] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[1], (
        'new width (nout) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[1]

    if index is None:
        index = np.random.randint(teacher_w1.shape[1], size=n)
    factors = np.bincount(index)[index] + 1.
    new_w1 = teacher_w1[:, index]
    new_b1 = teacher_b1[index]
    new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)

    # add small noise to break symmetry, so that student model will have full capacity later
    noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape) if add_noise else 0
    student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
    student_w2[index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def shallow_to_medium(shallow_model, medium_model, inference_data, verbose=False):
    """
    Assumes the weights have already been loaded in/ initialized.

    medium_model adds a:
     - Third convolutional block + batchnorm + ReLU
     - Widens the Fully Connected layer by doubling the amount of nodes.

    Performs all changes to the models in-place.
    :param shallow_model: keras.Model The trained smaller model.
    :param medium_model: keras.Model The un-trained medium model.
    :param inference_data: np.array Dataset to infer statistics on for BatchNorm layers.
    :param verbose: bool Whether to print out transfer progress.
    """
    # Link between the model's layer idx.
    medium_receives_shallow = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                               (6, 6), (7, 7), (11, 8), (13, 10), (14, 11)]

    for (medium_idx, shallow_idx) in medium_receives_shallow:
        if verbose:
            print('Set parameters of {} to {}'.format(medium_idx, shallow_idx))
            print(medium_model.layers[medium_idx], shallow_model.layers[shallow_idx])
        medium_model.layers[medium_idx].set_weights(shallow_model.layers[shallow_idx].get_weights())

    # Convolutional layers
    if verbose:
        print("Deepening Convolution 8")
    new_conv = medium_model.layers[8]
    new_conv.set_weights(identity_convolution(new_conv.get_weights()[0]))

    # BatchNorm layer following after a Convolutional layer.
    if verbose:
        print("Deepening BatchNormalization layer 9. Performing forward inference.")
    new_bn = medium_model.layers[9]
    new_bn.set_weights(identity_batchnorm(
        medium_model.layers[0].input, new_bn.input, inference_data, axes=(0, 1, 2), epsilon=new_bn.epsilon))

    # Fully Connected layer.
    # Note! The output of the model splits into two unique dense layers.
    if verbose:
        print('Widening final Dense layer.')
    shallow_dense = shallow_model.layers[9]
    shallow_following_dense_pi = shallow_model.layers[12]
    shallow_following_dense_v = shallow_model.layers[13]

    medium_dense = medium_model.layers[12]
    medium_following_dense_pi = medium_model.layers[15]
    medium_following_dense_v = medium_model.layers[16]

    # Specify a random mapping for the net2net widening operator such
    # that the mapping is equal for the pi and v output dense layers.
    n = medium_dense.get_weights()[0].shape[1] - shallow_dense.get_weights()[0].shape[1]
    index = np.random.randint(shallow_dense.get_weights()[0].shape[1], size=n)

    w1, b1, pi_w = wider2net_fc(shallow_dense.get_weights()[0], shallow_dense.get_weights()[1],
                                shallow_following_dense_pi.get_weights()[0], medium_dense.get_weights()[0].shape[1],
                                index=index)
    w2, b2, v_w = wider2net_fc(shallow_dense.get_weights()[0], shallow_dense.get_weights()[1],
                               shallow_following_dense_v.get_weights()[0], medium_dense.get_weights()[0].shape[1],
                               index=index)

    # Ensure that no coincidental changes have been made to the unchanged layer.
    assert (b1 == b2).all()
    assert (w1 == w2).all()

    medium_dense.set_weights([w1, b1])
    medium_following_dense_pi.set_weights([pi_w, medium_following_dense_pi.get_weights()[1]])
    medium_following_dense_v.set_weights([v_w, medium_following_dense_v.get_weights()[1]])

    if verbose:
        print("Done, medium model now has an equivalent mapping as the shallow model.")


def medium_to_deep(medium_model, deep_model, inference_data, verbose=False):
    """
    Assumes the weights have already been loaded in/ initialized.

    Performs all changes to the models in-place.
    :param medium_model: keras.Model The trained medium model.
    :param deep_model: keras.Model The un-trained deep model.
    :param inference_data: np.array Dataset to infer statistics on for BatchNorm layers.
    :param verbose: bool Whether to print out transfer progress.
    """
    # Link between the model's layer idx.
    deep_receives_medium = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                            (7, 7), (8, 8), (9, 9), (10, 10), (13, 10), (14, 11),
                            (16, 13), (17, 14), (21, 15), (22, 16)]

    for (deep_idx, medium_idx) in deep_receives_medium:
        if verbose:
            print('Set parameters of {} to {}'.format(deep_idx, medium_idx))
            print(deep_model.layers[deep_idx], medium_model.layers[medium_idx])
        deep_model.layers[deep_idx].set_weights(medium_model.layers[medium_idx].get_weights())

    # Convolutional layers
    if verbose:
        print("Deepening Convolution 11")
    new_conv = deep_model.layers[11]
    new_conv.set_weights(identity_convolution(new_conv.get_weights()[0]))

    # BatchNorm layer following after a Convolutional layer.
    if verbose:
        print("Deepening BatchNormalization 12. Performing forward inference.")
    new_bn = deep_model.layers[12]
    new_bn.set_weights(identity_batchnorm(
        deep_model.layers[0].input, new_bn.input, inference_data, axes=(0, 1, 2), epsilon=new_bn.epsilon))

    # Deepen the Dense layer first before widening it.
    if verbose:
        print("Deepening, then Widening final two Dense layers.")
    medium_dense = medium_model.layers[12]
    weights_deepened_dense = identity_fc(medium_dense.get_weights()[0])  # LINK: Deepened

    deep_dense = deep_model.layers[15]  # Widened
    deep_following_dense = deep_model.layers[18]  # LINK: Deepened

    widened_weights, widened_bias, deepened_weights = wider2net_fc(
        medium_dense.get_weights()[0], medium_dense.get_weights()[1], weights_deepened_dense,
        deep_dense.get_weights()[0].shape[1], add_noise=True)

    deep_dense.set_weights([widened_weights, widened_bias])
    deep_following_dense.set_weights([deepened_weights, deep_following_dense.get_weights()[1]])

    if verbose:
        print("Done, deep model now has an equivalent mapping as the medium model.")
