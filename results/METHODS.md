# HALT study, methods record

This file records how every number in the study was produced. It is written by `python -m experiments.methods` from the code and the saved runs, so it always matches the data. It is a record of facts to write a paper from, not paper text.

## 1. What the study tests

The proposal (Bhat, October 2026) states this hypothesis.

> This work hypothesizes that per-layer internal signals (weight change, gradient norm, weight norm, and fraction of zero outputs) will expose a broken layer within the first few epochs, before the accuracy curve shows anything wrong, without flagging healthy runs.

The study turns that into four measurable questions.

1. Does HALT flag broken runs, and does it flag them earlier than the accuracy curve?
2. When HALT flags a run, does it name the layer where the fault was planted and the right cause?
3. How often does HALT flag healthy runs it has never seen?
4. Do the same rules work on three different networks when only the healthy baselines are rebuilt?

Every broken run has a fault planted on purpose, so the true answer for each run is known.

## 2. Networks and data

| Network | Dataset | Task | Classes or vocabulary | Training examples | Validation examples | Parameters | Watched layers | ReLU layers | Epochs | Batch size | Steps per epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fully connected | Fashion-MNIST, 28x28 grayscale | image classification | 10 | 60,000 | 10,000 | 242,762 | 4 | 3 | 20 | 128 | 469 |
| Transformer | Tiny Shakespeare, characters | next character prediction | 65 | 1,003,854 | 512 | 112,577 | 11 | 2 | 15 | 32 | 100 |
| CNN | GTSRB traffic signs, 32x32 color | image classification | 43 | 26,640 | 4,000 | 41,659 | 5 | 4 | 15 | 128 | 209 |

For the transformer, training examples count characters in the training text and validation examples count fixed windows of 64 characters, so 512 windows give 32,768 scored next character predictions.

| Network | Layer | Module | Weight shape | Parameters | ReLU after it |
|---|---|---|---|---|---|
| Fully connected | fc1 | Linear | 256x784 | 200,960 | yes |
| Fully connected | fc2 | Linear | 128x256 | 32,896 | yes |
| Fully connected | fc3 | Linear | 64x128 | 8,256 | yes |
| Fully connected | fc4 | Linear | 10x64 | 650 | no |
| Transformer | tok | Embedding | 65x64 | 4,160 | no |
| Transformer | pos | Embedding | 64x64 | 4,096 | no |
| Transformer | b0.qkv | Linear | 192x64 | 12,480 | no |
| Transformer | b0.proj | Linear | 64x64 | 4,160 | no |
| Transformer | b0.ff1 | Linear | 256x64 | 16,640 | yes |
| Transformer | b0.ff2 | Linear | 64x256 | 16,448 | no |
| Transformer | b1.qkv | Linear | 192x64 | 12,480 | no |
| Transformer | b1.proj | Linear | 64x64 | 4,160 | no |
| Transformer | b1.ff1 | Linear | 256x64 | 16,640 | yes |
| Transformer | b1.ff2 | Linear | 64x256 | 16,448 | no |
| Transformer | head | Linear | 65x64 | 4,225 | no |
| CNN | conv1 | Conv2d | 8x3x3x3 | 224 | yes |
| CNN | conv2 | Conv2d | 16x8x3x3 | 1,168 | yes |
| CNN | conv3 | Conv2d | 32x16x3x3 | 4,640 | yes |
| CNN | fc1 | Linear | 64x512 | 32,832 | yes |
| CNN | fc2 | Linear | 43x64 | 2,795 | no |

Images are scaled to the range 0 to 1 and standardized with the mean and standard deviation of the training set. Fashion-MNIST uses one mean for all pixels and GTSRB uses one per color channel. GTSRB images are resized to 32 by 32 pixels with bilinear resampling. The GTSRB files come from the Hugging Face copy of the torchvision splits because the original host refused connections. Tiny Shakespeare is split into the first 90% of characters for training and the last 10% for validation, one token per character.

The validation set is each dataset's official test split. Nothing is tuned on it. It only feeds the accuracy and loss curves that HALT is compared against. The CNN is scored on a fixed random 4,000 image subset of the GTSRB test split, chosen once with seed 2, to save time.

## 3. How one training run works

1. Set the random seed, then build the network, so the seed fixes the starting weights.
2. Seed a separate generator with the same seed. It fixes the order of training batches.
3. Train with Adam at learning rate 0.001 and a cosine schedule that falls toward 0 over the run.
4. At the start of each epoch, switch on the planted fault if this is its onset epoch.
5. After every backward pass and before the optimizer step, HALT records each watched layer's gradient norm.
6. After every optimizer step, a frozen layer gets its old weights put back.
7. At the end of each epoch, score the validation set, then HALT measures every watched layer.
8. Every run trains for all its epochs even when HALT would have stopped it, so each saved run can be replayed through any detector afterward.
9. Save the run as JSON with every epoch's signals, accuracy, loss, and timing.

Each run uses 1 CPU thread and 5 runs train at once. Replaying a saved run through HALT gives the same result as running HALT live, which a test in `tests/test_replay.py` checks.

## 4. What HALT measures

For each watched layer, each epoch. W is the layer's weights and bias flattened into one vector.

| Signal | Formula | When it is measured |
|---|---|---|
| Weight change | ‖W_end − W_start‖ / ‖W_start‖ | Start and end of the epoch |
| Weight norm | ‖W_end‖ | End of the epoch |
| Gradient norm | mean over batches of ‖∂loss/∂W‖ | Every batch, after the backward pass |
| Zero activation fraction | share of units whose output is exactly 0 for every probe input | End of the epoch, on the probe batch |

‖·‖ is the square root of the sum of squared entries. The probe batch is the same 256 validation examples in every run, chosen once with seed 0. It is run in evaluation mode with no gradients. A unit is one neuron of a dense layer or one filter of a convolution. A filter counts as dead only when it outputs 0 at every position for every probe input.

## 5. The healthy range and the alarm rule

The healthy range is built only from calibration runs, seeds 0 to 19, 20 runs per network with no fault. For every layer, signal, and epoch it keeps the lowest and highest value across those runs. Broken runs never touch it.

A value counts as out of range when it is below low × (1 − m) or above high × (1 + m) + f, with margin m = 0.5 and floor f = 0.05. A value that is not a finite number is always out of range.

HALT raises an alarm when one signal of one layer stays out of range for p = 3 epochs in a row. If several layers reach that point in the same epoch, HALT reports the first one in the order data flows through the network.

The settings m = 0.5, f = 0.05, and p = 3 were taken from the HALT library's own tests and fixed before any broken run was trained. They were never tuned on broken runs. Section 11 shows how results change across other settings.

## 6. How HALT names a cause

The rules run in this order on the alarmed layer's signals at the alarm epoch.

| Order | Pattern | Cause named |
|---|---|---|
| 1 | any signal not a finite number | diverged |
| 2 | zero activation fraction above range | dead units |
| 3 | weight change below range and gradient norm inside or above range | frozen layer |
| 4 | weight change below range and gradient norm below range | vanishing gradient |
| 5 | weight norm above range | runaway weights |
| 6 | anything else | unrecognized |

## 7. Planted faults

Every fault goes into one layer of one run.

| Fault | How it is planted | Onset epoch | Where |
|---|---|---|---|
| Frozen layer | after every optimizer step the layer's weights and bias are put back, so gradients are computed but never applied | 1 | every watched layer |
| Dead units, 10%, 25%, 50% | that share of the layer's units, picked at random with seed + 7919, get incoming weights of 0 and a bias of -1.0, so their ReLU output is 0 for every input. Adam's running averages for those units are cleared | 1 | every layer followed by a ReLU |
| Late frozen layer | as above | 6 | every watched layer |
| Late dead units, 50% | as above | 6 | every layer followed by a ReLU |
| Cut gradient | the input to one module is detached, so every layer before it gets no gradient | 1 | CNN conv3 and fc1, Fully connected fc2 and fc4, Transformer blocks.1 and head |
| Learning rate jump | the learning rate is multiplied by 10 or by 100 | 6 | whole network |

The proposal's faults are the frozen layer and dead units planted at epoch 1. The others are extra tests in the more phase.

## 8. Runs and seeds

| Role | Seeds | Purpose |
|---|---|---|
| Calibration | 0 to 19 | build the healthy range, nothing else |
| Fresh healthy | 1000 to 1029, then 1030 to 1049 | measure false alarms |
| Broken, core phase | 1000 to 1004 | the proposal's faults |
| Broken, seeds phase | 1005 to 1009 | the proposal's faults on 5 more seeds |
| Broken, more phase | 1000 to 1004 | faults beyond the proposal |

Calibration seeds and evaluation seeds never overlap. Every broken run has a healthy twin, the fresh run with the same seed, which starts from the same weights and sees batches in the same order. The twin gives the accuracy each broken run would have reached without its fault.

| Network | Phase | Role | Planned runs | Finished runs |
|---|---|---|---|---|
| CNN | core | calibration | 20 | 20 |
| CNN | core | fault | 85 | 85 |
| CNN | core | fresh | 30 | 30 |
| CNN | more | fault | 65 | 0 |
| CNN | more | fresh | 20 | 0 |
| CNN | seeds | fault | 85 | 20 |
| Fully connected | core | calibration | 20 | 20 |
| Fully connected | core | fault | 65 | 65 |
| Fully connected | core | fresh | 30 | 30 |
| Fully connected | more | fault | 55 | 0 |
| Fully connected | more | fresh | 20 | 0 |
| Fully connected | seeds | fault | 65 | 65 |
| Transformer | core | calibration | 20 | 20 |
| Transformer | core | fault | 85 | 85 |
| Transformer | core | fresh | 30 | 30 |
| Transformer | more | fault | 85 | 0 |
| Transformer | more | fresh | 20 | 0 |
| Transformer | seeds | fault | 85 | 85 |

Planned runs in total 885. Finished when this file was written 555.

## 9. Detectors compared

Every detector sees the same saved runs and, where it needs one, the same calibration runs.

| Detector | Rule | Needs healthy runs | Defined |
|---|---|---|---|
| HALT | section 5, patience 3 | yes | before testing |
| Accuracy curve, 3 epochs | validation accuracy below the lowest calibration value for that epoch, 3 epochs in a row | yes | before testing |
| Accuracy curve, 1 epoch | the same with 1 epoch | yes | before testing |
| Loss curve, 3 epochs | validation loss above the highest calibration value for that epoch, 3 epochs in a row | yes | before testing |
| Zero update check | some watched layer's weight change is exactly 0 in an epoch | no | after the core results |
| Dead units check, 25%, 50%, 70%, 80% | some layer has at least that share of dead units | no | after the core results |
| Zero update or dead units 50% | either of the two checks above | no | after the core results |
| SageMaker DeadRelu default | the 50% dead units check above. SageMaker counts a ReLU as inactive when it is active less than 1% of the time, close to HALT's never active | no | after the core results |
| DeepDiagnosis Dead Node | some ReLU layer has more than 70% inactive units, the check in Wardat et al. (2022) | no | after the core results |
| wandb-callbacks default | some ReLU layer has at least 80% dead units, the default of the package's DeadReluCallback | no | after the core results |
| wandb-callbacks verbose | some ReLU layer has at least one dead unit, the package's verbose mode | no | after the core results |

The accuracy curve rule uses no margin, which makes it more sensitive than HALT's rule. The simple checks were added after the core results to test whether HALT does better than rules anyone could write by hand. They can only make HALT look the same or worse. Every dead unit check looks only at layers followed by a ReLU. The wandb-callbacks rules copy the dead ReLU warning of the package cited in the proposal, which counts a unit as dead when it never fires on the inputs it is given, the same definition HALT uses.

## 10. Outcome measures

| Measure | Definition |
|---|---|
| Caught | the detector fired at or after the fault's onset epoch |
| Alarm epoch | the epoch the detector fired |
| Right layer | HALT named the planted layer. For a cut gradient, any layer before the cut |
| Right cause | frozen layer for frozen, dead units for dead, vanishing gradient for a cut |
| False alarm | the detector fired at any epoch of a fresh healthy run |
| Gap | accuracy curve epoch minus HALT epoch, with never counted as one epoch past the end |
| Compute saved | share of the run's epochs left after HALT's alarm, 0 when not caught |
| Silent | final validation accuracy at or above the lowest calibration final accuracy |
| Accuracy cost | the healthy twin's final accuracy minus the broken run's final accuracy |
| Harm group | under 0.5 pts, 0.5 to 2 pts, 2 pts or more |
| Overhead | time spent inside HALT divided by the whole run's time, on fresh healthy runs |

## 11. Statistics

Rates come with 95% Wilson score intervals. Runs that share a seed share starting weights and batch order, and runs that share a fault and layer share the same break, so single runs are not independent. The main tests average the gap within each seed and within each fault condition first, then run a one sided Wilcoxon signed rank test on those averages. The 95% interval for the mean gap comes from a bootstrap with 10,000 draws that resamples whole seeds within each network. Run by run tests are reported for reference only.

Three sweeps replay the same saved runs. The settings sweep tries margins 0, 0.1, 0.25, 0.5, and 0.75, floors 0, 0.02, 0.05, and 0.1, and patience 1 to 5. The calibration sweep builds the healthy range from 2, 3, 5, 10, 15, and 20 calibration runs, 20 random draws each. The signal ablation turns each signal off, or uses one signal alone.

## 12. Software, hardware, and compute

Python 3.10.16, PyTorch 2.2.2, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.10.0. All runs trained on one laptop CPU, Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz, with 1 thread per run. The laptop was often thermally throttled and shared with other work, so timing numbers are rough.

| Network | Runs | Training hours | Hours spent inside HALT | Mean minutes per run |
|---|---|---|---|---|
| Fully connected | 180 | 13.9 | 1.7 | 4.6 |
| Transformer | 220 | 35.2 | 2.0 | 9.6 |
| CNN | 155 | 30.3 | 0.7 | 11.7 |
| All networks | 555 | 79.4 | 4.4 |  |

## 13. Choices and changes made during the study

| What | When | Effect on HALT's results |
|---|---|---|
| Datasets, networks, epochs, and batch sizes were chosen for this study | before training | none intended |
| The accuracy curve's alarm was defined as below every healthy run for 3 epochs in a row | before testing | this is how the proposal's comparison was made measurable |
| Dead units get a bias of -1.0 instead of the library default of -10 | after a pilot run, before the study runs | -10 inflated the weight norm far past normal, which would have handed HALT an easy signal. -1 makes dead units harder for HALT to catch |
| The CNN was made smaller, the CNN and transformer were cut to 15 epochs, the transformer to 100 steps per epoch, and the CNN scored on 4,000 test images | before any CNN or transformer run | none intended, done to fit the laptop's compute |
| Adam's running averages are cleared for killed units | after the core fully connected runs, before any late fault ran | affects only late dead unit faults, which would otherwise come back to life |
| Seed and fault condition level statistics became the main tests | after the fully connected core results | more conservative than run by run tests |
| Simple fixed checks and the harm split were added | after the core results | can only make HALT look the same or worse |

## 14. Where the code differs from the proposal's wording

| Proposal says | The study does | Why |
|---|---|---|
| the zero activation fraction is how much of a layer's outputs are exactly zero | the share of a layer's units that output exactly zero for every probe input, so a unit counts only when it is dead for all inputs | this is how the HALT library's signal was written. The proposal's sentence that dead units read near one fits both versions. The paper should describe the one in the code |
| HALT stops the run once a fault is confirmed | every run trains to the end and HALT's alarm is found by replaying the saved run | so every detector can be scored on the same full runs. Replay matches live HALT, which `tests/test_replay.py` checks, and `experiments/demo.py` shows a live run being stopped |
| thresholds are set just outside the healthy range | the range is widened by a margin of 0.5 and a floor of 0.05 before a value counts as out | thresholds right at the range's edge flag many healthy runs, which section 11's sweep shows |
| a layer is flagged after numerous consecutive epochs | 3 consecutive epochs | taken from the HALT library's tests |
| the CNN is trained first, then the fully connected network and the transformer | the fully connected network ran first, then the transformer, then the CNN | to fit the laptop's schedule. The rules were fixed before any of them ran |
| the frozen layer is tested first, followed by dead neurons | both faults ran in the same phase | order does not change a replayed result |
| frozen layers and dead neurons | also late faults, cut gradients, learning rate jumps, 5 more seeds, and extra baselines | extra tests beyond the proposal, reported apart from its main results |

## 15. Where the data is

| File | Contents |
|---|---|
| results/runs/<network>.jsonl.gz | every saved run, one JSON object per line |
| results/tables/runs.csv | one scored row per fresh and broken run |
| results/tables/epochs.csv | accuracy and loss for every run and epoch |
| results/tables/signals.csv.gz | every signal for every run, epoch, and layer |
| results/tables/detectors.csv | every detector on the same runs |
| results/HALT_data.xlsx | every table in one workbook |
| results/REPORT.md | the results with figures |
