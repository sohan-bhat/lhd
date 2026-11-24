# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR During the development of models at larger scales we see that the majority of resources are expended during the training lifecycle. But why? Well, as models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data which creates a compounding effect. Now since most runs, epochs, are experimental, teams often train many times before they ship a viable model. Additively, this makes one kind of bug expensive to deal with: silent failures. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. Unfortunately, we only find out if that specific epoch was a success or not after the time and compute are used up. The proposed Layer Health Diagnostics (LHD) harness monitors each layer during training and intervenes in these runs as necessary.

We started this work after one such failure. While training SignNet, a small CNN for German traffic signs (GTSRB), we got 76% accuracy and spent weeks on tuning before we found the cause, which was a single convolutional layer that we had left out of the optimizer, so the weights of that layer were never updated. Once we fixed that we got 87% with the same network, and since we saw no error in either run, the only sign of the bug was a lower accuracy than we should have had.

## Layer Health Diagnostics

To tell a healthy layer from a broken one during a run, we log four signals for each layer after every epoch, namely weight change, gradient norm, weight norm, and the fraction of dead units, which are units with an output of zero for every input. We then compare each signal with the range we measured for that signal in healthy runs of the same network, and since we set that range from healthy runs only, we never need a broken run for the setup. If a signal is outside that range for three straight epochs, we stop the run and report the layer along with a likely cause ([Figure 1](#figure-1)).

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change (left) is 0 for the whole run while validation accuracy (right) is still in the healthy range.
<!-- /figure -->

## Results

To test this, we used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare, and trained 885 runs in total. For each network we first set the ranges from 20 healthy runs, and after that we broke 470 runs on purpose, kept 150 fresh healthy runs to count false alarms, and used the rest for other faults.

In each broken run we planted one fault, which is either a frozen layer, as in SignNet, or dead units, where we forced 10%, 25%, or 50% of a layer's units to an output of zero. For comparison, we also watched the accuracy curve of every run, and we flagged a run there when the validation accuracy was below the lowest healthy value for three straight epochs.

| | LHD | Accuracy curve |
|---|---|---|
| Broken runs detected, out of 470 | 411 | 166 |
| Detected, out of the 105 runs with a loss of 2 points or more | 94 | 60 |
| Healthy runs flagged, out of 150 | 3 | 7 |

Overall, we detected 87% of the broken runs with LHD, usually at epoch 3, and in 98% of those we also named the layer where we had planted the fault. Additionally, on the 162 runs that we detected both ways, the alarm from LHD was 4.6 epochs earlier on average than the one from the accuracy curve.

However, in 223 of the 470 broken runs we lost under half a point of accuracy, which is too little to see in the curve. For that reason, in the second row of the table we count only the 105 runs with a loss of 2 points or more.

## Limits

Unfortunately, for frozen layers a much simpler test is better than LHD. When we only tested each layer for a weight change of exactly zero, we found all 200 frozen layers at epoch 1 with no false alarms, which is two epochs before LHD, so we do not need LHD for the original SignNet bug.

But why can't we use a fixed threshold in the same way for dead units? Well, we also see dead units in healthy networks, and at very different rates in different layers, so no single threshold is right for every layer. With LHD we detected 78% of dead unit faults with 3 false alarms, whereas with the default rule of SageMaker Debugger we detected 40% of them with 22 false alarms.

Still, we missed faults with LHD, and with only 10% of a layer's units killed we detected 57% of runs. For instance, in the dense layer of the CNN between 36% and 70% of units were already dead in healthy runs, so even with 50% of those units killed we detected only 1 of 10 runs.

As for the 3 false alarms, two of them were in a CNN layer with only 16 filters, where a single dead filter is already over the threshold.

Finally, we planted every fault here, and we made each one visible in the signals we log, so we have not yet tried LHD on real bugs. Additionally, we never put a fault in the data, such as flipped labels, so we do not know whether we would detect one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and they are shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset, whereas in LHD we measure each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd), and every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, and drafted this post and parts of the full PDF. I directed the project, decided what to test and change, and I'm responsible for all of its content.
