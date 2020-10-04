from utils.loss_utils import scalar_loss, scale_gradient

import tensorflow as tf


class MuZeroNeuralNet:
    """
    This class specifies the base NeuralNet class. To define your own neural
    network, subclass this class and implement the functions below. The neural
    network does not consider the current player, and instead only deals with
    the canonical form of the board.

    See othello/NNet.py for an example implementation.
    """

    def __init__(self, game, net_args, builder):
        self.net_args = net_args
        self.neural_net = builder(game, net_args)

        self.optimizer = tf.optimizers.Adam(self.net_args.lr)

    def loss_function(self, observations, actions, target_vs, target_rs, target_pis, sample_weights):

        @tf.function
        def loss():
            total_loss = tf.constant(0, dtype=tf.float32)

            # Root inference
            s = self.neural_net.encoder(observations)
            pi_0, v_0 = self.neural_net.predictor(s[..., 0])

            # Collect predictions of the form: [w_i * 1 / K, v, r, pi] for each forward step k...K
            predictions = [(sample_weights, v_0, None, pi_0)]
            for t in range(actions.shape[1]):  # Shape (batch_size, K, action_size)
                r, s = self.neural_net.dynamics([s[..., 0], actions[:, t, :]])
                pi, v = self.neural_net.predictor(s[..., 0])

                predictions.append((sample_weights / len(actions), v, r, pi))
                s = scale_gradient(s, 1 / 2)

            for t in range(len(predictions)):  # Length = 1 + K (root + hypothetical forward steps)
                gradient_scale, vs, rs, pis = predictions[t]
                t_vs, t_rs, t_pis = target_vs[t, ...], target_rs[t, ...], target_pis[t, ...]

                r_loss = scalar_loss(rs, t_rs) if t > 0 else 0
                v_loss = scalar_loss(vs, t_vs)
                pi_loss = scalar_loss(pis, t_pis)

                step_loss = r_loss + v_loss + pi_loss
                total_loss += tf.reduce_sum(scale_gradient(step_loss, gradient_scale))

            return total_loss
        return loss

    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        pass

    def get_variables(self):
        """
        Yield a list of all trainable variables within the model

        Returns:
            variable_list: A list of all tf.Variables within the entire MuZero model.
        """
        pass

    def encode(self, observations):
        """
        Input:
            observations: A history of observations of an environment (in canonical form).

        Returns:
            s_0: A neural encoding of the environment.
        """
        pass

    def forward(self, latent_state, action):
        """
        Input:
            latent_state: A neural encoding of the environment at step k: s_k.
            action: A (encoded) action to perform on the latent state

        Returns:
            r: The immediate predicted reward of the environment
            s_(k+1): A new 'latent_state' resulting from performing the 'action' in
                the latent_state.
        """
        pass

    def predict(self, latent_state):
        """
        Input:
            latent_state: A neural encoding of the environment's.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass
