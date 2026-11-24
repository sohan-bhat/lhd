# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR During the development of models at larger scales we see that the majority of resources are expended during the training lifecycle. But why? Well, as models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data which creates a compounding effect. Now since most runs, epochs, are experimental, teams often train many times before they ship a viable model. Additively, this makes one kind of bug expensive to deal with: silent failures. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. Unfortunately, we only find out if that specific epoch was a success or not after the time and compute are used up. The proposed Layer Health Diagnostics (LHD) harness monitors each layer during training and intervenes in these runs as necessary.

This work started from one such failure in SignNet, a small CNN that classifies German traffic signs (GTSRB). It sat at 76% accuracy through weeks of tuning before the cause turned out to be a single convolutional layer that had been left out of the optimizer, so its weights stayed frozen for the whole run. Once that was fixed the same network reached 87%, and since training had finished with no error either way, the only sign of the bug was an accuracy lower than the network could have reached.

## Layer Health Diagnostics

To tell a healthy layer from a broken one while the run is still going, LHD logs four signals for each layer after every epoch, namely weight change, gradient norm, weight norm, and the fraction of dead units, which are units that output zero for every input. Each signal is then compared with the range it stays in during healthy runs of the same network, and since that range is set from healthy runs only, no broken run is ever needed to set it up. If a signal stays outside its range for three straight epochs, the run is stopped and LHD reports which layer it was along with a likely cause ([Figure 1](#figure-1)).

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change (left) stays at 0 while validation accuracy (right) still looks healthy.
<!-- /figure -->

## Results

To test this, we used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare, and trained 885 runs in total. For each network the ranges were first set from 20 healthy runs, and after that 470 runs were broken on purpose, 150 fresh healthy runs were kept to count false alarms, and the rest were used for other faults.

Each broken run carries one fault, which is either a frozen layer, as in SignNet, or dead units, where 10%, 25%, or 50% of a layer's units are forced to output zero. For comparison, we also watched the accuracy curve of every run, and a run was flagged there when its validation accuracy stayed below the lowest healthy value for three straight epochs.

| | LHD | Accuracy curve |
|---|---|---|
| Broken runs detected, out of 470 | 411 | 166 |
| Detected, out of the 105 runs that lost 2 points or more | 94 | 60 |
| Healthy runs flagged, out of 150 | 3 | 7 |

Overall, LHD detected 87% of the broken runs, usually at epoch 3, and in 98% of those it also named the layer where the fault had been planted. Additionally, on the 162 runs that both methods detected, LHD was 4.6 epochs earlier on average than the accuracy curve.

However, in 223 of the 470 broken runs the network lost under half a point of accuracy, which is too little to show in the curve. For that reason, the second row of the table counts only the 105 runs that lost 2 points or more.

## Limits

Unfortunately, for frozen layers a much simpler test does better than LHD. When we only tested whether each layer's weights had moved at all, all 200 frozen layers were found at epoch 1 with no false alarms, which is two epochs before LHD, so the original SignNet bug does not need LHD at all.

But why can't a fixed threshold do the same for dead units? Well, units also die in healthy networks, and they die at very different rates in different layers, which means that no single threshold fits every layer. With LHD, 78% of dead unit faults were detected with 3 false alarms, whereas SageMaker Debugger's default rule detected 40% of them with 22 false alarms.

Still, LHD misses faults, and with only 10% of a layer's units killed it detected 57% of runs. For instance, in the CNN's dense layer between 36% and 70% of units were already dead in healthy runs, so even with 50% of its units killed only 1 of 10 runs was detected.

As for the 3 false alarms, two of them came from a CNN layer with only 16 filters, where a single dead filter is already over the threshold.

Finally, every fault here was planted, and each one was made so that it shows in the signals LHD logs, which means that LHD has not yet been tried on real bugs. Additionally, no fault was ever put in the data, such as flipped labels, so we do not know whether LHD would detect one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and they are shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset, whereas LHD measures each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd), and every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, and drafted this post and parts of the full PDF. I directed the project, decided what to test and change, and I'm responsible for all of its content.
