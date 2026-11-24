# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR Training takes the majority of the compute spent on a large model. As models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data, which creates a compounding effect. Most runs are also experimental, so a team often trains many times before shipping a viable model. Together, these facts make one kind of bug expensive to deal with, namely the silent failure. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. The failure of such a run shows only after the time and compute are used up. Layer Health Diagnostics, or LHD, checks each layer during training and stops a run when a layer breaks. Across 885 training runs on three networks, we detected 97.7% of the 470 planted faults with LHD, between 95.3% and 100% on each network, including all 105 with a loss of 2 points or more in accuracy, and we flagged 6 of the 150 healthy runs.

I began this work after one such silent failure occurred in a project of my own. I obtained 76% accuracy with SignNet, a small convolutional network for the GTSRB dataset of German traffic signs, and spent weeks on tuning before identifying the cause. A single convolutional layer had been omitted from the optimizer, so the weights of that layer were never updated. Correcting the omission raised the accuracy of the same network to 87%, and because neither run raised an error, the only indicator of the bug was an accuracy 11 points below the corrected result.

## Layer Health Diagnostics

LHD records five signals for each layer after every epoch, namely weight change, gradient norm, weight norm, the dead-unit fraction, and the always-dead fraction. A dead unit has an output of zero for every input, and an always-dead unit has been dead before training and at every epoch since. Each signal is compared with a healthy range, which is the range measured for that signal in healthy runs of the same network. Every healthy range is calibrated on healthy runs only, so calibration never requires a faulty run. A detection occurs when a signal remains outside the healthy range for three consecutive epochs, after which the run is terminated and the layer is reported together with a probable cause, as [Figure 1](#figure-1) shows.

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen-layer fault, weight change on the left remains at 0 for the entire run while validation accuracy on the right stays within the healthy range.
<!-- /figure -->

## Results

The evaluation used three networks: a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare. I trained 885 runs in total and calibrated the healthy ranges of each network on 20 healthy runs. A fault was then injected into 470 runs, 150 held-out healthy runs were reserved to measure false positives, and the remaining runs were used to test other kinds of faults.

Each faulty run contained exactly one injected fault, either a frozen-layer fault or a dead-unit fault. A frozen-layer fault reproduces the SignNet bug, in which one layer receives no weight updates, and a dead-unit fault forces 10%, 25%, or 50% of the units in one layer to an output of zero. The accuracy baseline follows the validation accuracy curve of every run and records a detection when validation accuracy falls below the lowest healthy value for three consecutive epochs.

I first conducted the entire evaluation with a four-signal variant that omits the always-dead fraction. After examining the false negatives of that variant, I added the always-dead fraction as the fifth signal, set the rule for that signal on runs that were never scored, and then scored every run once more. The table below therefore shows both variants side by side with the accuracy baseline.

| | LHD | Four-signal variant | Accuracy baseline |
|---|---|---|---|
| Faulty runs detected, out of 470 | 459 | 411 | 166 |
| Detected, out of the 105 runs with an accuracy loss of 2 points or more | 105 | 94 | 60 |
| False positives, out of 150 healthy runs | 6 | 3 | 7 |

The table shows that LHD detected 459 of the 470 faulty runs, or 98%, with a typical detection epoch of 3. In 98% of those detections, the reported layer was the layer that contained the injected fault. On the 165 runs detected by both methods, the detection from LHD preceded the detection from the accuracy baseline by 4.9 epochs on average. The cost of the fifth signal appears in the last row of the table, where LHD produced 6 false positives on the 150 healthy runs and the four-signal variant produced only 3.

In 223 of the 470 faulty runs, the accuracy loss was under half a point, which is too small to observe in the accuracy curve. The second row of the table therefore counts only the 105 runs with an accuracy loss of 2 points or more.

## Limitations

The results of this study carry six limitations that later work should address. First, a much simpler test performs better than LHD for frozen-layer faults. A zero-change test, which checks each layer only for a weight change of exactly zero, identified all 200 frozen layers at epoch 1 with no false positives, which is two epochs earlier than LHD. LHD is therefore unnecessary for detecting the original SignNet bug.

The same shortcut does not transfer to dead-unit faults, because dead units also occur in healthy networks at rates that differ widely between layers, so no single threshold suits every layer. LHD detected 96% of dead-unit faults with 6 false positives, whereas the default rule of SageMaker Debugger detected 40% of those faults with 22 false positives.

Second, LHD still produced 11 false negatives, and in every one of those runs only 10% of the units in a layer had been disabled. In the dense layer of the CNN, between 5% and 19% of units were always dead even in healthy runs, so LHD detected only 2 of the 10 runs with 10% of the units in that layer disabled.

Third, LHD produced 6 false positives on the 150 healthy runs. Three originate from the first four signals, and two of those three occurred in a CNN layer with only 16 filters, where a single dead filter already exceeds the healthy range. The other three originate from the always-dead fraction in the dense layer of the CNN, and in two of those runs fewer units were always dead than in any of the 20 calibration runs, which does not constitute a fault.

Fourth, every fault in this study was injected and made visible in the recorded signals, so LHD has not yet been evaluated on naturally occurring bugs. Fifth, I added the always-dead fraction only after examining the false negatives of the four-signal variant. Every unit in a dead-unit fault was disabled before training, which is exactly the case that this signal was designed for, so the share of this gain that would carry over to other bugs remains unknown. For example, neither variant detected any of the 5 runs in which half of the units in the dense layer of the CNN were disabled at epoch 6. Finally, no fault was injected into the training data, so label errors and similar data faults remain untested.

## Related work

Comparable signals are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and the dashboard of [wandb](https://www.wandb.com/) displays the same signals during training. The first three systems use preset thresholds, whereas LHD calibrates the healthy range of each layer on healthy runs of the same network.

## Code and data

[The repo](https://github.com/sohan-bhat/lhd) contains the code, every saved run, and every table of this study.

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library for the first four signals (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, designed and wrote the fifth signal after I asked for a higher detection rate, and drafted this post. I directed the project, decided what to test and change, and I'm responsible for all of its content.
