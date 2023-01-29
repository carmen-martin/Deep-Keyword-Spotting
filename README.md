## Small Footprint Keyword Spotting with different ML architectures

This repository contains the implementation in  <tt>Tensorflow 2.11.0</tt> of different models for the KWS task. 

**CNN with dropout(0.2)** - 4 classes (3 keywords + 1) - test acc: 97.72%

- Total params: 1,394,592
- Trainable params: 1,394,216
- Non-trainable params: 376

**CRNN** - 4 classes (3 keywords + 1) - test acc: 96.23%

- Total params: 608,484
- Trainable params: 608,484
- Non-trainable params: 0

**Autoencoder** (+ SVM) - 4 classes (3 keywords + 1) - test acc: 92.36% - code dim: 12

- Total params: 1,531,381
- Trainable params: 1,531,381
- Non-trainable params: 0

**PCA + SVM** - 257 dim - test acc: 88.89%

- Trainable params: 66,049

**PCA + SVM** - 25 dim - test acc: 86.35%

- Trainable params: 625
