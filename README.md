# Basic MuZero and AlphaZero in Tensorflow 
We provide a readable, commented, well documented, and conceptually easy implementation of the AlphaZero and MuZero algorithms based on the popular AlphaZero-General implementation. 
Our implementation extends AlphaZero to work with single player domains, like its successor MuZero.
The codebase provides a modular framework to design your own AlphaZero and MuZero models and an API to pit the two algorithms against each other. 
This API also allows MuZero agents to more strongly rely on their learned model during interaction with the environment; the programmer can specify the sparsity of observations to a *learned* MuZero agent during a trial. 
Our interface also provides sufficient abstraction to extend the MuZero or AlphaZero algorithm for research purposes.

**beta phase**: Most of the codebase is done regarding development, we are currently working on finishing up this project and making it easily available for other users.

## Example Results
This codebase was designed for a Masters Course at Leiden University, we utilized the code to create visualizations of the learned MDP model within MuZero. 
We did this exclusively for MountainCar, the visualization tool can be viewed here: https://kaesve.nl/projects/cartpole-inspector/#/.
Requisite data for viewing learned models can be accessed here TODO or generated using the jupyter notebook here TODO. 
The figure below illustrates one of our learned models:

<object data="publish/figures/MC_l4kl_MDPAbstractionCombined.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="publish/figures/MC_l4kl_MDPAbstractionCombined.pdf">
        <p>This browser does not support PDF embedding. </p>
    </embed>
</object>

We quantified the efficacy of our MuZero and AlphaZero implementations also on the CartPole environment over numerous hyperparameters.

No boardgames were tested as computation time quickly became an issue for us, even on smaller boardsizes.

Our paper can be read *here*. TODO

### Minimal requirements
* Python 3.7+
 - tensorflow
 - keras standalone (until tensorflow 2.3 is available on anaconda windows)

#### Tested Versions
* Python 3.7.9
 - tensorflow 2.1.0
 - keras 2.3.1
 
## Our Contributions
There are already a variety of MuZero and AlphaZero implementations available:
- AlphaZero-General (any framework; sequential): https://github.com/suragnair/alpha-zero-general
- MuZero-General (Pytorch; parallelized): https://github.com/werner-duvaud/muzero-general
- MuZero in Tensorflow (Tensorflow; sequential): https://github.com/johan-gras/MuZero

Our implementation is intended to be both pedagogical and functional. 
This means that we focus on documentation, elegance, and clarity of the code.
For this exact reason have we omitted parallelization during training of the agents.

## References
- Schrittwieser, Julian et al. (Feb. 21, 2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”. [cs, stat]. arXiv:1911.08265
- Silver, David et al. (Dec. 2018). “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play”. In:Science 362.6419, pp. 1140–1144.DOI:10.1126/science.aar6404



