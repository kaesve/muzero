# Basic MuZero and AlphaZero in Tensorflow 
We provide a readable, commented,well documented, and conceptually easy implementation of the AlphaZero and MuZero algorithms. 
Our implementation of AlphaZero is an adaptation of the original algorithm that is also able to learn effectively in single player domains, like its successors MuZero.
The codebase provides a modular framework to design your own AlphaZero and MuZero models and an API to pit the two algorithms against each other.
Our interface also provides sufficient abstraction to extend the MuZero or AlphaZero algorithm for research purposes.

**work in progress**: Note that the codebase is currently still in development, this may imply that the code is subject to refactoring, change, etc.

## Links
- Schrittwieser, Julian et al. (Feb. 21, 2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”. [cs, stat]. arXiv:1911.08265
- Silver, David et al. (Dec. 2018). “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play”. In:Science 362.6419, pp. 1140–1144.DOI:10.1126/science.aar6404
- Inspiration of our codebase: https://github.com/suragnair/alpha-zero-general
- Parallelized implementation of MuZero in PyTorch: https://github.com/werner-duvaud/muzero-general

## Minimal requirements
* Python 3.7+
 - tensorflow
 - keras standalone (until tensorflow 2.3 is available on anaconda windows)

### Tested Versions
* Python 3.7.9
 - tensorflow 2.1.0
 - keras 2.3.1
 
## TODOs
 - Running a large number of empirical tests. Including comparing AlphaZero against MuZero on Atari and Hex.
 - Multiprocessing for self-play and pitting (no concurrency of weight-updates and self-play!).
 - Minimal user/ developer documentation for adapting our code.
 - (Continuous Alpha/ MuZero)
 - (Add unique environments.)
 