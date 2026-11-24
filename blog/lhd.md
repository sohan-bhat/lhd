# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR Training takes the majority of the compute spent on a large model. As models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data, which creates a compounding effect. Most runs are also experimental, so a team often trains many times before shipping a viable model. Together, these facts make one kind of bug expensive to deal with, namely the silent failure. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. The failure of such a run shows only after the time and compute are used up. Layer Health Diagnostics, or LHD, checks each layer during training and stops a run when a layer breaks. Across 885 training runs on three networks, we detected 97.7% of the 470 planted faults with LHD, between 95.3% and 100% on each network, including all 105 with a loss of 2 points or more in accuracy, and we flagged 6 of the 150 healthy runs.

I began this work after encountering one such failure in a project of my own. While training SignNet, a small convolutional network for German traffic signs from the GTSRB dataset, I obtained 76% accuracy and spent weeks on tuning before identifying the cause. A single convolutional layer had been omitted from the optimizer, so the weights of that layer were never updated. After correcting the omission I obtained 87% accuracy with the same network, and because neither run raised an error, the only indicator of the bug was an accuracy lower than the network could reach.

## Layer Health Diagnostics

To distinguish a healthy layer from a faulty one during a run, I record five signals for each layer after every epoch. The signals are weight change, gradient norm, weight norm, the fraction of dead units, and the fraction of always dead units. A dead unit produces an output of zero for every input, and an always dead unit was dead before training and at every epoch since. I then compare each signal with the range measured for that signal in healthy runs of the same network, and because that range is derived from healthy runs only, the setup never requires a faulty run. If a signal remains outside that range for three consecutive epochs, I terminate the run and report the layer together with a probable cause, as [Figure 1](#figure-1) shows.

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change on the left remains at 0 for the entire run while validation accuracy on the right stays within the healthy range.
<!-- /figure -->

## Results

To evaluate the method, I used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare, and trained 885 runs in total. For each network I first derived the ranges from 20 healthy runs, then injected a fault into 470 runs, reserved 150 fresh healthy runs to count false alarms, and used the remainder for other faults.

Each faulty run contained one injected fault, either a frozen layer, as in SignNet, or dead units, where 10%, 25%, or 50% of a layer's units were forced to an output of zero. As a comparison, I also monitored the accuracy curve of every run and flagged a run when the validation accuracy fell below the lowest healthy value for three consecutive epochs.

I first conducted the entire evaluation with only the first four signals. After examining the faulty runs that this version missed, I added the fifth signal, set the rule for that signal on runs that were never scored, and then scored every run once more. For that reason, both versions appear side by side in the table.

| | LHD | Four signals | Accuracy curve |
|---|---|---|---|
| Faulty runs detected, out of 470 | 459 | 411 | 166 |
| Detected, out of the 105 runs with a loss of 2 points or more | 105 | 94 | 60 |
| Healthy runs flagged, out of 150 | 6 | 3 | 7 |

Overall, LHD detected 98% of the faulty runs, usually at epoch 3, and in 98% of those detections the reported layer was the layer containing the injected fault. On the 165 runs detected by both methods, the alarm from LHD preceded the alarm from the accuracy curve by 4.9 epochs on average. The cost of the fifth signal appears in the last row of the table, because the five signal version flagged 6 of the 150 healthy runs, whereas the four signal version flagged only 3.

However, in 223 of the 470 faulty runs the loss in accuracy was under half a point, which is too small to observe in the curve. For that reason, the second row of the table counts only the 105 runs with a loss of 2 points or more.

## Limitations

For frozen layers, a much simpler test performs better than LHD does. When I tested each layer only for a weight change of exactly zero, the test identified all 200 frozen layers at epoch 1 with no false alarms, which is two epochs earlier than LHD, so LHD is unnecessary for the original SignNet bug.

A fixed threshold is not equally effective for dead units, because dead units also occur in healthy networks, and at very different rates in different layers, so no single threshold suits every layer. LHD detected 96% of dead unit faults with 6 false alarms, whereas the default rule of SageMaker Debugger detected 40% of those faults with 22 false alarms.

LHD still missed 11 faulty runs, and in every one of those runs only 10% of a layer's units had been disabled. For instance, in the dense layer of the CNN between 5% and 19% of units were always dead even in healthy runs, so with 10% of the units in that layer disabled, LHD detected only 2 of 10 runs.

Of the 6 false alarms, three originate from the first four signals, and two of those occurred in a CNN layer with only 16 filters, where a single dead filter already exceeds the threshold. The other three originate from the fifth signal, all in the dense layer of the CNN, and in two of those the run was flagged only because fewer units were always dead than in any of the 20 healthy runs behind the range, which does not constitute a fault.

Finally, I injected every fault in this study and made each one visible in the recorded signals, so LHD has not yet been evaluated on naturally occurring bugs. I also added the fifth signal only after examining the runs that the four signal version missed, and because every unit was disabled before training, which is exactly the kind of dead unit that the fifth signal targets, the share of this gain that would carry over to other bugs remains unknown. For example, when half of the units in the dense layer of the CNN were disabled at epoch 6, neither version detected any of the 5 runs. In addition, no fault was ever placed in the data, such as flipped labels, so the ability of LHD to detect such a fault remains untested.

## Related work

Comparable signals are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and [wandb](https://www.wandb.com/) displays the same signals on a dashboard. In the first three systems the thresholds are preset, whereas LHD derives the range of each layer from healthy runs of the same network.

## Code and data

The code, every saved run, and every table are available in [the repo](https://github.com/sohan-bhat/lhd).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library for the first four signals (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, designed and wrote the fifth signal after I asked for a higher detection rate, and drafted this post. I directed the project, decided what to test and change, and I'm responsible for all of its content.
