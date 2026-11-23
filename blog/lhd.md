# Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR I built LHD (Layer Health Diagnostics) to find neural network training runs that go wrong with no error. I broke 470 runs on purpose and caught 411 of them with it, or 87%. However, for one kind of fault a simpler test is better.

While training SignNet, a small CNN for German traffic signs (GTSRB), I got 76% accuracy and spent weeks tuning it. In the end, I found that I had left one convolutional layer out of the optimizer, so it stayed frozen, and after I fixed that I got 87%.

Training had finished with no error, yet I still got a worse model, and that is a silent failure. In this post, I show how I find these failures early.

## Layer Health Diagnostics

After every epoch I log four signals for each layer, namely weight change, gradient norm, weight norm, and the fraction of dead units. Next, I compare each signal with its range in healthy runs of the same network, and I set that range from healthy runs only. If a signal then stays outside its range for three straight epochs, I stop the run and report the layer and a likely cause ([Figure 1](#figure-1)).

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change (left) stays at 0 while validation accuracy (right) still looks healthy.
<!-- /figure -->

## Results

I used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare. In total I trained 885 runs. For each network I first set the ranges from 20 healthy runs. After that, I broke 470 runs, kept 150 fresh healthy runs to count false alarms, and used the rest for other faults.

Each broken run has one fault, either a frozen layer (my SignNet bug) or dead units, where I forced 10%, 25%, or 50% of a layer's units to output zero.

For comparison, I also watched the accuracy curve, and I flagged a run when its validation accuracy stayed below the lowest healthy value for three straight epochs.

| | LHD | Accuracy curve |
|---|---|---|
| Broken runs caught, out of 470 | 411 | 166 |
| Caught, out of the 105 runs that lost 2 points or more | 94 | 60 |
| Healthy runs flagged, out of 150 | 3 | 7 |

With LHD I caught 87% of the broken runs, usually at epoch 3, and in 98% of them I got the planted layer. Likewise, on the 162 runs I caught both ways, LHD was 4.6 epochs earlier on average.

However, in 223 of the 470 broken runs I lost under half a point of accuracy, which is too little to see in the curve, so in the second row I count only the runs with real damage.

## Limits

For frozen layers a simpler test is better. I tested whether each layer's weights moved, and I caught 200 of 200 frozen layers at epoch 1 with no false alarms, two epochs before LHD. So for my original bug I don't need LHD.

For dead units, on the other hand, LHD is better. Units also die in healthy networks, so I can't use one threshold for every layer. With LHD I caught 78% of dead unit faults with 3 false alarms, but with SageMaker Debugger's default rule I caught 40% with 22.

Still, I missed faults. With 10% of a layer's units killed, I caught 57% of runs. Likewise, in the CNN's dense layer between 36% and 70% of units were already dead in healthy runs, so with 50% killed I caught 1 of 10 runs.

As for the 3 false alarms, two were in a CNN layer with 16 filters, where one dead filter is over the threshold.

Finally, I planted every fault and made each one to show in the signals I log, so I haven't tried LHD on real bugs. I also never put a fault in the data, like flipped labels, so I don't know if I'd catch one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset, whereas in LHD I measure each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd), and every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, and drafted this post and parts of the full PDF. I directed the project, decided what to test and change, and I'm responsible for all of its content.
