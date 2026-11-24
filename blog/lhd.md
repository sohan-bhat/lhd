# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR During the development of models at larger scales, we see that the majority of compute is expended during the training lifecycle. But why? Well, as models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data, which creates a compounding effect. Now, since most runs are experimental, teams often train many times before they ship a viable model. Together, this makes one kind of bug expensive to deal with: silent failures. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. Unfortunately, we only find out if that run was a success after the time and compute are used up. Layer Health Diagnostics (LHD) monitors each layer during training and stops these runs when a layer breaks. Across 885 training runs on three networks, we detected 97.7% of the 470 planted faults with LHD, between 95.3% and 100% on each network, including all 105 with a loss of 2 points or more in accuracy, and we flagged 6 of the 150 healthy runs.

We started this work after one such failure. While training SignNet, a small CNN for German traffic signs (GTSRB), we got 76% accuracy and spent weeks on tuning before we found the cause, which was a single convolutional layer that we had left out of the optimizer, so the weights of that layer were never updated. Once we fixed that we got 87% with the same network, and since we saw no error in either run, the only sign of the bug was a lower accuracy than we should have had.

## Layer Health Diagnostics

To tell a healthy layer from a broken one during a run, we log five signals for each layer after every epoch, namely weight change, gradient norm, weight norm, the fraction of dead units, which are units with an output of zero for every input, and the fraction of always dead units, which are units that were dead before training and at every epoch since. We then compare each signal with the range we measured for that signal in healthy runs of the same network, and since we set that range from healthy runs only, we never need a broken run for the setup. If a signal is outside that range for three straight epochs, we stop the run and report the layer along with a likely cause ([Figure 1](#figure-1)).

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change (left) is 0 for the whole run while validation accuracy (right) is still in the healthy range.
<!-- /figure -->

## Results

To test this, we used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare, and trained 885 runs in total. For each network we first set the ranges from 20 healthy runs, and after that we broke 470 runs on purpose, kept 150 fresh healthy runs to count false alarms, and used the rest for other faults.

In each broken run we planted one fault, which is either a frozen layer, as in SignNet, or dead units, where we forced 10%, 25%, or 50% of a layer's units to an output of zero. For comparison, we also watched the accuracy curve of every run, and we flagged a run there when the validation accuracy was below the lowest healthy value for three straight epochs.

We first ran this whole test with only the first four signals, and after we saw which broken runs we had missed, we added the fifth signal, set the rule for that signal on runs that we never scored, and then scored every run one more time. For that reason, we show both versions in the table.

| | LHD | Four signals | Accuracy curve |
|---|---|---|---|
| Broken runs detected, out of 470 | 459 | 411 | 166 |
| Detected, out of the 105 runs with a loss of 2 points or more | 105 | 94 | 60 |
| Healthy runs flagged, out of 150 | 6 | 3 | 7 |

Overall, we detected 98% of the broken runs with LHD, usually at epoch 3, and in 98% of those we also named the layer where we had planted the fault. Additionally, on the 165 runs that we detected both ways, the alarm from LHD was 4.9 epochs earlier on average than the one from the accuracy curve. The cost of the fifth signal is in the last row of the table, since with five signals we flagged 6 of the 150 healthy runs, where with four we had flagged only 3.

However, in 223 of the 470 broken runs we lost under half a point of accuracy, which is too little to see in the curve. For that reason, in the second row of the table we count only the 105 runs with a loss of 2 points or more.

## Limits

Unfortunately, for frozen layers a much simpler test is better than LHD. When we only tested each layer for a weight change of exactly zero, we found all 200 frozen layers at epoch 1 with no false alarms, which is two epochs before LHD, so we do not need LHD for the original SignNet bug.

But why can't we use a fixed threshold in the same way for dead units? Well, we also see dead units in healthy networks, and at very different rates in different layers, so no single threshold is right for every layer. With LHD we detected 96% of dead unit faults with 6 false alarms, whereas with the default rule of SageMaker Debugger we detected 40% of them with 22 false alarms.

Still, we missed 11 broken runs with LHD, and in every one of them we had killed only 10% of a layer's units. For instance, in the dense layer of the CNN between 5% and 19% of units were always dead even in healthy runs, so with 10% of the units in that layer killed we detected only 2 of 10 runs.

As for the 6 false alarms, three are from the first four signals, and two of those were in a CNN layer with only 16 filters, where a single dead filter is already over the threshold. The other three are from the fifth signal, all in the dense layer of the CNN, and in two of them we flagged the run only because fewer units were always dead than in any of the 20 healthy runs behind the range, which is not a fault at all.

Finally, we planted every fault here, and we made each one visible in the signals we log, so we have not yet tried LHD on real bugs. We also added the fifth signal only after we had seen the runs that we missed with four, and since we killed every unit before training, which is exactly the kind of dead unit that the fifth signal was built for, we do not know how much of this gain we would keep on other bugs. For example, when we killed half of the units in the dense layer of the CNN at epoch 6, we detected 0 of 5 runs with either version. Additionally, we never put a fault in the data, such as flipped labels, so we do not know whether we would detect one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and they are shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset, whereas in LHD we measure each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd), and every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library for the first four signals (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, designed and wrote the fifth signal after I asked for a higher detection rate, and drafted this post and parts of the full PDF. I directed the project, decided what to test and change, and I'm responsible for all of its content.
