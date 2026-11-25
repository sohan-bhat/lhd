# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR Training takes the majority of the compute spent on a large model. As models advance, the compute needed for training increases exponentially, because each generation has more parameters and learns from more data, which creates a compounding effect. Most runs are also experimental, so a team often trains many times before shipping a viable model. Together, these facts make one kind of bug expensive to deal with, namely the silent failure. For instance, a layer in a convolutional neural network may stop learning while the outputs look normal. The failure of such a run shows only after the time and compute are used up. Layer Health Diagnostics, or LHD, checks each layer during training and stops a run when a layer breaks. Across 885 training runs on three networks, we detected 97.7% of the 470 planted faults with LHD, between 95.3% and 100% on each network, including all 105 with a loss of 2 points or more in accuracy, and we flagged 6 of the 150 healthy runs.

I began this work after one such silent failure occurred in a project of my own. I obtained 76% accuracy with SignNet, a small convolutional network for the GTSRB dataset of German traffic signs, and spent weeks on tuning before identifying the cause. A single convolutional layer had been omitted from the optimizer, so the weights of that layer were never updated. After the omission was corrected, the accuracy of the same network rose to 87%, and because no error occurred in either run, the only indicator of the bug was an accuracy 11 points below the corrected result.

## Layer Health Diagnostics

LHD consists of four stages, which are recording, calibration, detection, and diagnosis. During a run, the five signals in the table below are recorded for each layer after every epoch. A dead unit has an output of zero for every input of a fixed probe batch of 256 examples, and an always-dead unit has been dead before training and at every epoch since.

| Signal | Definition | Fault indicated by the signal |
|---|---|---|
| Weight change | Movement of the weights of a layer relative to their size | A layer whose weights are never updated |
| Gradient norm | Size of the gradient of a layer for one batch | A layer with a gradient close to zero |
| Weight norm | Overall size of the weights of a layer | Weights far larger than in healthy runs |
| Dead-unit fraction | Share of units with an output of zero for every probe input | Units with zero output that were active earlier |
| Always-dead fraction | Share of units that were dead at every measurement | Units with zero output since before training |

During calibration, a healthy range is measured for every signal, layer, and epoch on 20 healthy runs of the same network. The lowest and the highest value across those runs form the range, which is then widened so that ordinary noise does not cause a detection. The lower bound is multiplied by 0.5, and the upper bound is multiplied by 1.5 and raised by 0.05. A lower bound above zero therefore remains above zero, so a weight change of exactly zero always falls outside the healthy range. Because calibration is performed on healthy runs only, no faulty run is needed for the setup.

A detection occurs when a signal remains outside the healthy range for three consecutive epochs. The run is then terminated, and the layer is reported together with a probable cause. The cause is determined from the pattern of the signals at the detection, with the patterns checked in the order of the table below.

| Signal pattern at the detection | Reported cause |
|---|---|
| Any signal is not a finite number | Diverged training |
| Dead-unit fraction or always-dead fraction above the healthy range | Dead units |
| Weight change below the range, gradient norm inside or above the range | Frozen layer |
| Weight change and gradient norm both below the range | Vanishing gradient |
| Weight norm above the range | Runaway weights |
| Any other pattern | Unrecognized cause |

One run with a frozen-layer fault in the second convolutional layer of the CNN appears in [Figure 1](#figure-1). The weight change of that layer is exactly 0 from the first epoch, which lies below the healthy range, while validation accuracy stays inside the healthy range for the entire run. A detection was raised at epoch 3 with a frozen layer reported as the cause, whereas no detection was recorded for the accuracy baseline in 15 epochs.

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen-layer fault, weight change on the left remains at 0 for the entire run while validation accuracy on the right stays within the healthy range.
<!-- /figure -->

## Experimental setup

The evaluation was conducted on the three networks in the table below. Two networks are image classifiers with dense and convolutional layers, and the third is a next character predictor with attention layers. I trained 885 runs in total and calibrated the healthy ranges of each network on 20 healthy runs. A fault was then injected into 470 runs, 50 held-out healthy runs per network were reserved to measure false positives, and the remaining runs were used to test other kinds of faults.

| Network | Dataset | Parameters | Watched layers | Epochs per run |
|---|---|---|---|---|
| Fully connected network | Fashion-MNIST | 242,762 | 4 | 20 |
| CNN | GTSRB traffic signs | 41,659 | 5 | 15 |
| Character transformer | Tiny Shakespeare | 112,577 | 11 | 15 |

Each faulty run contained exactly one injected fault, either a frozen-layer fault or a dead-unit fault. In a frozen-layer fault, the weights of one layer are never updated, as in the SignNet bug. In a dead-unit fault, 10%, 25%, or 50% of the units in one layer are forced to an output of zero. For the accuracy baseline, the validation accuracy curve of every run is tracked, and a detection is recorded when validation accuracy falls below the lowest healthy value for three consecutive epochs.

I first conducted the entire evaluation with a four-signal variant without the always-dead fraction. After examining the false negatives of that variant, I added the always-dead fraction as the fifth signal, set the rule for that signal on runs that were never scored, and then scored every run once more. Both variants therefore appear side by side with the accuracy baseline in the first results table.

## Results

| | LHD | Four-signal variant | Accuracy baseline |
|---|---|---|---|
| Faulty runs detected, out of 470 | 459 | 411 | 166 |
| Detected, out of the 105 runs with an accuracy loss of 2 points or more | 105 | 94 | 60 |
| False positives, out of 150 healthy runs | 6 | 3 | 7 |

According to the table, 459 of the 470 faulty runs, or 98%, were detected with LHD, against 166 with the accuracy baseline. The median detection epoch was 3 on every network, which is one fifth of a 15 epoch run and less than one sixth of a 20 epoch run. On the 165 runs detected by both methods, the detection from LHD preceded the detection from the accuracy baseline by 4.9 epochs on average.

Of the 166 faulty runs detected with the accuracy baseline, 165 were also detected with LHD. A total of 294 faulty runs were therefore detected with LHD alone, and 1 run was detected with the accuracy baseline alone. The difference is clearest on the character transformer, where the accuracy loss stayed under 2 points in all 170 faulty runs and all 170 were still detected with LHD.

| Network | Faulty runs detected | Correct layer reported | False positives |
|---|---|---|---|
| Fully connected network | 127 of 130 | 127 of 127 | 1 of 50 |
| CNN | 162 of 170 | 155 of 162 | 5 of 50 |
| Character transformer | 170 of 170 | 170 of 170 | 0 of 50 |

The detection rate was similar across all three networks, with 97.7% on the fully connected network, 95.3% on the CNN, and 100% on the character transformer. In 452 of the 459 detections, or 98%, the reported layer was the layer that contained the injected fault. All 7 detections with a different reported layer were frozen-layer faults in the CNN.

| Injected fault | Runs | Detected by LHD |
|---|---|---|
| Frozen-layer fault | 200 | 200 |
| Dead-unit fault with 50% of the units disabled | 90 | 90 |
| Dead-unit fault with 25% of the units disabled | 90 | 90 |
| Dead-unit fault with 10% of the units disabled | 90 | 79 |

Every frozen-layer fault and every dead-unit fault with 25% or 50% of the units in a layer disabled was detected with LHD. For dead-unit faults overall, 259 of 270 runs, or 96%, were detected with LHD, with 6 false positives. With the default rule of SageMaker Debugger, 40% of the same faults were detected, with 22 false positives. Dead units also occur in healthy networks at rates that differ widely between layers, so no single threshold is correct for every layer.

The second row of the first table is restricted to the faults with the largest effect on the final model. In 223 of the 470 faulty runs, the accuracy loss was under half a point, which is too small to observe in the accuracy curve. In 105 runs the accuracy loss was 2 points or more, and all 105 of those runs were detected with LHD, against 94 with the four-signal variant and 60 with the accuracy baseline. Of those 105 runs, 95 came from the CNN and 10 came from the fully connected network.

The difference between the two variants comes from the fifth signal, with 48 additional detections at the cost of 3 additional false positives. Recording the four original signals took 2.2% of the run time on the CNN, 5.9% on the character transformer, and 11.6% on the fully connected network.

## Limitations

The results reported in this study are subject to five limitations. First, the full method is not needed for frozen-layer faults, including the original SignNet bug. With a zero-change test, in which each layer is checked only for a weight change of exactly zero, all 200 frozen layers were identified at epoch 1 with no false positives, two epochs ahead of LHD. The same 200 faults were also detected with LHD, so the zero-change test is simply the faster option for this one fault type, while the calibrated healthy ranges remain necessary for dead-unit faults.

Second, the smallest dead-unit fault lies below the sensitivity of LHD. All 11 false negatives occurred with 10% of the units in a layer disabled, and the accuracy loss stayed under 2 points in every one of those runs. Eight of the 11 occurred in the dense layer of the CNN, because between 5% and 19% of the units in that layer were always dead even in healthy runs, and only 2 of the 10 runs with that fault there were detected.

Third, 6 false positives occurred with LHD on the 150 healthy runs, one fewer than with the accuracy baseline. Three originate from the first four signals and three from the always-dead fraction. Two of the six occurred in a CNN layer with only 16 filters, where a single dead filter already exceeds the healthy range. Two more were flagged because fewer units were always dead than in any of the 20 calibration runs, which does not constitute a fault.

Fourth, every fault in this study was injected and made visible in the recorded signals, so an evaluation on naturally occurring bugs is the next step. I added the always-dead fraction after examining the false negatives of the four-signal variant, and every unit in a dead-unit fault was disabled before training, which is the case that this signal was designed for. The gain from the fifth signal may therefore be smaller on other bugs. For example, none of the 5 runs in which half of the units in the dense layer of the CNN were disabled at epoch 6 was detected with either variant.

Fifth, no fault was injected into the training data, so label errors and similar data faults remain untested.

## Related work

Comparable signals are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and the same signals are displayed on the dashboard of [wandb](https://www.wandb.com/) during training. In the first three systems the thresholds are preset, whereas in LHD the healthy range of each layer is calibrated on healthy runs of the same network.

## Code and data

[The repo](https://github.com/sohan-bhat/lhd) contains the code, every saved run, and every table of this study.

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library for the first four signals (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, designed and wrote the fifth signal after I asked for a higher detection rate, and drafted this post. I directed the project, decided what to test and change, and I'm responsible for all of its content.
