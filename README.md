This repository holds code for my project "Evaluating the reliability of current state-of-the-art action recognition models under impact of adversarial examples".

To setup the original Temporal Segment Network, please refer to the author's repository ([link](https://github.com/yjxiong/temporal-segment-networks)).

Two AE crafting algorithms are implemented to attack TSN. They are Fast Gradient Sign Method and Basic Iterative Fast Gradient Sign Method proposed in this paper ([link](https://arxiv.org/abs/1611.01236)).

To attack by using FGSM algorithm, run the following script
```
python tools/ae_attack_tsn.py
```

To attack by using Basic Iter FGSM algorithm, run the following script
```
python tools/ae_attack_iter_tsn.py
```
