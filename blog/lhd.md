# Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

**TL;DR** I built LHD (Layer Health Diagnostics) to catch neural network training runs that go wrong with no error. I caught 411 of 470 broken runs with it. For one kind of fault a simpler test is better.

I trained SignNet, a small CNN for German traffic signs (GTSRB). I got 76% accuracy and spent weeks tuning it. I had left one convolutional layer out of the optimizer, so it stayed frozen. I fixed that and got 87%.

I finished training with no error and still got a worse model. That is a silent failure.

## Layer Health Diagnostics

After every epoch I log four signals for each layer. They are weight change, gradient norm, weight norm, and the fraction of dead units. I compare each with its range in healthy runs of the same network, and I set that range from healthy runs only. If a signal stays outside its range for three straight epochs, I stop the run and report the layer and a likely cause ([Figure 1](#figure-1)).

<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

**Figure 1.** In a CNN run with a frozen layer, weight change (left) stays at 0 while validation accuracy (right) still looks healthy.

## Results

I used three networks. They were a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare.

I ran 885 runs. For each network I set the ranges from 20 healthy runs. I broke 470 runs, kept 150 fresh healthy runs to count false alarms, and used the rest for other faults.

I put one fault in each broken run. It was a frozen layer (my SignNet bug) or dead units, where I forced 10%, 25%, or 50% of a layer's units to output zero.

I compared LHD with the accuracy curve. With the curve, I flag a run when its validation accuracy stays below the lowest healthy value for three straight epochs.

| | LHD | Accuracy curve |
|---|---|---|
| Broken runs caught, out of 470 | 411 | 166 |
| Caught, out of the 105 runs that lost 2 points or more | 94 | 60 |
| Healthy runs flagged, out of 150 | 3 | 7 |

With LHD I caught 87% of the broken runs, usually at epoch 3, and in 98% of them I got the planted layer. On the 162 runs I caught both ways, LHD was 4.6 epochs earlier on average.

In 223 of the 470 broken runs I lost under half a point of accuracy, too little to see in the curve. In the second row I count only the runs with real damage.

## Limits

For frozen layers a simpler test is better. I tested whether each layer's weights moved and caught 200 of 200 frozen layers at epoch 1 with no false alarms, two epochs before LHD. For my original bug I don't need LHD.

LHD is better for dead units. Units also die in healthy networks, so I can't use one threshold for every layer. With LHD I caught 78% of dead unit faults with 3 false alarms. With SageMaker Debugger's default rule I caught 40% with 22.

I still missed faults. With 10% of a layer's units killed, I caught 57% of runs. In the CNN's dense layer, between 36% and 70% of units were already dead in healthy runs, so with 50% killed I caught 1 of 10 runs.

Two of the 3 false alarms were in a CNN layer with 16 filters, where one dead filter is over the threshold.

I planted every fault. I made them to show in the signals I log, and I haven't tried LHD on real bugs. I never put a fault in the data, like flipped labels, so I don't know if I'd catch one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset. In LHD I measure each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd). Every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments in an earlier proposal, and wrote the core LHD library (signals, detector, and diagnosis), with Claude, made by Anthropic, providing function outlines and docstrings. Claude wrote the experiment scripts that trained and scored the networks, ran the training runs, wrote the analysis and figure code, and drafted parts of the full PDF, including the results and related work. Claude also wrote the first draft of this post. I directed the project, decided what to test and change, and I'm responsible for all of its content.
