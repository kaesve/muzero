"""
Defines a generic implementation of MuZeroNeuralNet to handle data for training, and performing standard inference.

This class should be able to handle most MuZero neural network models on most environment.

Notes:
 -  Base implementation done 15/11/2020
 -  Documentation 16/11/2020
"""
import numpy as np
import sys
import typing

from tensorflow import GradientTape

from utils.loss_utils import support_to_scalar, scalar_to_support, cast_to_tensor
from MuZero.MuNeuralNet import MuZeroNeuralNet
import Agents

from utils.storage import DotDict

sys.path.append('../../..')


class DefaultMuZero(MuZeroNeuralNet):
    """
    This class implements the MuZeroNeuralNet class. The implemented methods perform data manipulation to train
    the MuZero agent and return inferred variables given the necessary inputs.
    """

    def __init__(self, game, net_args: DotDict, architecture: str) -> None:
        """
        Initialize the MuZero Neural Network. Selects a Neural Network class constructor by an 'architecture' string.
        See Agents/MuZeroNetworks for all available neural architectures, or for defining new architectures.
        :param game: Implementation of base Game class for environment logic.
        :param net_args: DotDict Data structure that contains all neural network arguments as object attributes.
        :param architecture: str Neural network architecture to build in the super class.
        """
        super().__init__(game, net_args, Agents.MuZeroNetworks[architecture])
        self.action_size = game.getActionSize()
        self.architecture = architecture

    def get_variables(self) -> typing.List:
        """ Get all trainable parameters defined by the neural network """
        parts = (self.neural_net.encoder, self.neural_net.predictor, self.neural_net.dynamics)
        return [v for v_list in map(lambda n: n.weights, parts) for v in v_list]

    def train(self, examples: typing.List) -> float:
        """
        This function trains the neural network with data gathered from self-play.

        The examples data list is unpacked and formatted to the correct dimensions for the MuZero unrolling/
        loss computation. The resulting, formatted, data (np.ndarray) are cast to tf.Tensors before being
        passed to the MuNeuralNet loss_function class. This loss function call is done inside a tf.GradientTape
        to observe the gradients of all defined variables within the tf.graph. Based on the recorded gradient
        we perform one weight update using the optimizer defined in the super class. Returned loss values are
        additionally sent to the Monitor class for logging.

        :param examples: a list of training examples of length batch_size and the form:
                         (observation_trajectories, action_trajectories, targets, loss_scales).
                         Dimensions should be of the form:
                         observations: batch_size x width x height x (depth * time)
                         actions: batch_size x k x |action-space|
                         forward_observations: batch_size x k x width x height x (depth * time)
                         target_vs, target_rs: batch_size x k
                         target_pi: batch_size x k x |action-space|
                         sample_weight: batch_size x 1
        """
        # Unpack and transform data for loss computation.
        observations, actions, targets, forward_observations, sample_weight = list(zip(*examples))

        forward_observations = np.asarray(forward_observations)

        actions, sample_weight = np.asarray(actions), np.asarray(sample_weight)

        # Unpack and encode targets. Value target shapes are of the form [time, batch_size, categories]
        target_vs, target_rs, target_pis = list(map(np.asarray, zip(*targets)))

        target_vs = np.asarray([scalar_to_support(target_vs[:, t], self.net_args.support_size)
                                for t in range(target_vs.shape[-1])])
        target_rs = np.asarray([scalar_to_support(target_rs[:, t], self.net_args.support_size)
                                for t in range(target_rs.shape[-1])])
        target_pis = np.swapaxes(target_pis, 0, 1)

        # Pack formatted inputs as tensors.
        data = [cast_to_tensor(x) for x in [observations, actions, target_vs, target_rs,
                                            target_pis, forward_observations, sample_weight]]

        # Track the gradient through unrolling and loss computation and perform an optimization step.
        with GradientTape() as tape:
            loss, step_losses = self.loss_function(*data)

        grads = tape.gradient(loss, self.get_variables())
        self.optimizer.apply_gradients(zip(grads, self.get_variables()), name=f'MuZeroDefault_{self.architecture}')

        # Logging
        self.monitor.log(loss / len(examples), "total loss")
        for k, step_loss in enumerate(step_losses):
            self.monitor.log_recurrent_losses(k, *step_loss)

        self.steps += 1

    def initial_inference(self, observations: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, float]:
        """
        Combines the prediction and representation models into one call. This reduces
        overhead and results in a significant speed up.

        The observation array is padded with a batch-size dimension of length 1. The inferred state value is
        cast from its distributional bins into a scalar.

        :param observations: A game specific (stacked) tensor of observations of the environment at step t: o_t.
        :return: A tuple with predictions of the following form:
            s_(0): The root 'latent_state' produced by the representation function
            pi: a policy vector for the provided state - a numpy array of length |action_space|.
            v: a float that gives the state value estimate of the provided state.
        """
        # Pad batch dimension
        observations = observations[np.newaxis, ...]

        s_0, pi, v = self.neural_net.forward.predict(observations)

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)

        return s_0[0], pi[0], np.ndarray.item(v_real)

    def recurrent_inference(self, latent_state: np.ndarray, action: int) -> typing.Tuple[float, np.ndarray,
                                                                                         np.ndarray, float]:
        """
        Combines the prediction and dynamics models into one call. This reduces overhead.

        Integer actions are encoded to one-hot-encoded vectors. Both the latent state and action vector are
        padded with a batch-size dimensions of length 1. Inferred reward and state value values are
        cast from their distributional bins into scalars.

        :param latent_state: A neural encoding of the environment at step k: s_k.
        :param action: A (encoded) action to perform on the latent state
        :return: A tuple with predictions of the following form:
            r: The immediate predicted reward of the environment.
            s_(k+1): A new 'latent_state' resulting from performing the 'action' in the latent_state.
            pi: a policy vector for the provided state - a numpy array of length |action_space|.
            v: a float that gives the state value estimate of the provided state.
        """
        # One hot encode integer actions.
        a_plane = np.zeros(self.action_size)
        a_plane[action] = 1

        # Pad batch dimension
        latent_state = latent_state[np.newaxis, ...]
        a_plane = a_plane[np.newaxis, ...]

        r, s_next, pi, v = self.neural_net.recurrent.predict([latent_state, a_plane])

        # Cast bins to scalar
        v_real = support_to_scalar(v, self.net_args.support_size)
        r_real = support_to_scalar(r, self.net_args.support_size)

        return np.ndarray.item(r_real), s_next[0], pi[0], np.ndarray.item(v_real)
