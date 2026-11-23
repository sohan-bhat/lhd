# LHD: Early Detection of Silent Training Failures in Neural Networks

### Author

Sohan Bhat

### Published

November 2026

TL;DR I built LHD (Layer Health Diagnostics) to find neural network training runs that go wrong with no error, and with it I caught 411 of the 470 runs that I broke on purpose, or 87%. However, I also found that for one of the two kinds of fault in those runs, a much simpler test is better than LHD.

While I was training SignNet, a small CNN that classifies German traffic signs (GTSRB), I got 76% accuracy and then spent weeks tuning it before I found the actual cause. It turned out that I had left one convolutional layer out of the optimizer, so its weights stayed frozen for the whole run, and after I fixed that, the same network reached 87%.

Now since training had finished with no error and I still ended up with a worse model, I call this a silent failure. Unfortunately, we usually only find out about one after the time and compute are already used up, so in this post I show how I find these failures early, while the run is still going.

## Layer Health Diagnostics

After every epoch I log four signals for each layer, namely weight change, gradient norm, weight norm, and the fraction of dead units, which are units that output zero for every input. Next, I compare each signal with the range it stays in during healthy runs of the same network, and I set that range from healthy runs only, so I never need a broken run to set it up. If a signal then stays outside its range for three straight epochs, I stop the run and report which layer it was along with a likely cause ([Figure 1](#figure-1)).

<!-- figure -->
<a id="figure-1"></a>

![Weight change and validation accuracy for one CNN run with a frozen layer](img/how_lhd_works.png)

Figure 1. In a CNN run with a frozen layer, weight change (left) stays at 0 while validation accuracy (right) still looks healthy.
<!-- /figure -->

## Results

I used three networks, a fully connected network on Fashion-MNIST, a CNN on GTSRB, and a character transformer on Tiny Shakespeare, and in total I trained 885 runs. For each network I first set the ranges from 20 healthy runs, and after that I broke 470 runs, kept 150 fresh healthy runs to count false alarms, and used the rest for other faults.

Each broken run has one fault, which is either a frozen layer (my SignNet bug) or dead units, where I forced 10%, 25%, or 50% of a layer's units to output zero. For comparison, I also watched the accuracy curve of every run, and I flagged a run when its validation accuracy stayed below the lowest healthy value for three straight epochs.

| | LHD | Accuracy curve |
|---|---|---|
| Broken runs caught, out of 470 | 411 | 166 |
| Caught, out of the 105 runs that lost 2 points or more | 94 | 60 |
| Healthy runs flagged, out of 150 | 3 | 7 |

With LHD I caught 87% of the broken runs, usually at epoch 3, and in 98% of the runs I caught, the layer I reported was the one I had planted the fault in. Additionally, on the 162 runs that I caught both ways, LHD was 4.6 epochs earlier on average than the accuracy curve.

However, in 223 of the 470 broken runs I lost under half a point of accuracy, which is too little to see in the curve. So in the second row of the table I count only the 105 runs that lost 2 points or more, because those are the runs with real damage.

## Limits

For frozen layers a simpler test is better, because when I only tested whether each layer's weights had moved at all, I caught 200 of 200 frozen layers at epoch 1 with no false alarms, which is two epochs before LHD. So for my original bug, which was exactly this kind of frozen layer, I don't need LHD at all.

But why can't a fixed threshold do the same for dead units? Well, units also die in healthy networks, and they die at very different rates in different layers, so I can't use one threshold for every layer. With LHD I caught 78% of dead unit faults with 3 false alarms, whereas with SageMaker Debugger's default rule I caught 40% of them with 22 false alarms.

Unfortunately, I still missed faults, since with 10% of a layer's units killed I caught only 57% of runs. In the CNN's dense layer, for instance, between 36% and 70% of units were already dead in healthy runs, so even with 50% of its units killed I caught 1 of 10 runs.

As for the 3 false alarms, two of them were in a CNN layer with only 16 filters, where a single dead filter is already over the threshold.

Finally, I planted every fault myself and made each one so that it shows in the signals I log, which means I haven't tried LHD on real bugs yet. I also never put a fault in the data, such as flipped labels, so I don't know whether I would catch one.

## Related work

Signals like these are already used in [AutoTrainer](https://doi.org/10.1109/ICSE43902.2021.00043), [DeepDiagnosis](https://doi.org/10.1145/3510003.3510071), and [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/debugger-built-in-rules.html), and they are shown on a dashboard in [wandb](https://www.wandb.com/). In the first three the thresholds are preset, whereas in LHD I measure each layer's range from healthy runs of the same network.

## Code and data

The code and every saved run are in [the repo](https://github.com/sohan-bhat/lhd), and every table is in [the full PDF](https://github.com/sohan-bhat/lhd/blob/main/paper/lhd.pdf).

## Who did what

I came up with the idea, designed the experiments, and wrote the core LHD library (signals, detector, and diagnosis) from Claude's function outlines and docstrings. Claude, made by Anthropic, wrote the experiment, analysis, and figure code, ran the training runs, and drafted this post and parts of the full PDF. I directed the project, decided what to test and change, and I'm responsible for all of its content.
