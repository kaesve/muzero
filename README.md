# Basic MuZero and AlphaZero in Tensorflow 
We provide a readable, commented, well documented, and conceptually easy implementation of the AlphaZero and MuZero algorithms. 
Our implementation of AlphaZero is an adaptation of the original algorithm that is also able to learn effectively in single player domains, like its successor MuZero.
The codebase provides a modular framework to design your own AlphaZero and MuZero models and an API to pit the two algorithms against each other.
Our interface also provides sufficient abstraction to extend the MuZero or AlphaZero algorithm for research purposes.

**alpha phase**: Most of the codebase is done regarding development, however we may still decide to perform some refactoring, organization, additions, etc.

## Minimal requirements
* Python 3.7+
 - tensorflow
 - keras standalone (until tensorflow 2.3 is available on anaconda windows)

### Tested Versions
* Python 3.7.9
 - tensorflow 2.1.0
 - keras 2.3.1
 
## TODOs
 - Multiprocessing for self-play and pitting (no concurrency of weight-updates and self-play!).
 - Minimal user/ developer documentation for adapting our code.


## Links
- Schrittwieser, Julian et al. (Feb. 21, 2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”. [cs, stat]. arXiv:1911.08265
- Silver, David et al. (Dec. 2018). “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play”. In:Science 362.6419, pp. 1140–1144.DOI:10.1126/science.aar6404
- Inspiration of our codebase: https://github.com/suragnair/alpha-zero-general
- Parallelized implementation of MuZero in PyTorch: https://github.com/werner-duvaud/muzero-general

