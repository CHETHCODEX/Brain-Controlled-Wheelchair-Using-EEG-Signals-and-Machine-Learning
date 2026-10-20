# S001 Model Benchmarks

Date: 2026-10-20. This is the first real-data benchmark, not a successful
completion of the HLD accuracy requirements or a clinical validation.

## Data And Preprocessing

- PhysioNet EEGMMIDB v1.0.0, participant S001, runs 04/08/12 and 06/10/14.
- Downloaded through the official public AWS mirror; each EDF matched the
  mirror's published SHA-256 manifest before being accepted.
- Sampling rate 160 Hz; channels C3, Cz, C4, FC3, FC4, CP3, CP4, in that order.
- Zero-phase 60 Hz notch followed by 4th-order Butterworth 8-30 Hz bandpass.
- Thirty valid epochs per run, 180 in total, each shaped (7, 400).
- Epochs begin one second after a cue and last 2.5 seconds; models use the
  first 320 samples, matching the two-second replay buffer.
- Raw EDF files are unchanged. No ICA or automatic amplitude rejection is used.

The four-state class order is REST, TURN_LEFT, TURN_RIGHT, FORWARD, with
90, 23, 22, and 45 trials respectively. Bilateral fists and feet imagery are
merged into FORWARD. The binary task uses only the 45 left/right trials.

## Evaluation

CSP uses four components and Ledoit-Wolf regularization. LDA uses automatic
shrinkage; SVM uses an RBF kernel, C=1, and three-fold probability calibration.
These are the existing baseline configurations, not hyperparameters chosen
after inspecting S001 test-fold results.

Entire recording runs are held out with GroupKFold. CSP is fitted only on
each training fold. Binary evaluation uses three folds; four-state evaluation
uses five folds across six runs. Saved JSON reports include class counts,
confusion matrices, per-fold accuracy, and explicit training/validation run IDs.
The saved inference model is then refitted on all available task trials.

The EEGNet comparator uses the same input windows and outer run assignments.
It has eight temporal filters (kernel 32), spatial depth multiplier two,
16 separable-convolution features (temporal kernel 16), and dropout 0.25.
Per-channel mean and standard deviation are fitted on training-fold samples
only. Training uses Adam, learning rate 0.001, 50 fixed epochs, batch size 16,
unweighted cross entropy, and gradient norm clipping at one. CPU fitting uses
one thread, deterministic algorithms, seed 42 plus fold index, and seed 42
for the final refit. No early stopping or held-out-run epoch selection is used.
The tested runtime is PyTorch 2.14.1+cpu on Python 3.12.10. Bitwise reproducibility
is tested within this environment, not promised across library versions or hardware.

| Task | Model | Accuracy | Balanced Accuracy | Majority-Class Accuracy |
| --- | --- | ---: | ---: | ---: |
| Left/right | CSP + LDA | 71.11% | 71.44% | 51.11% |
| Left/right | CSP + SVM | 68.89% | 68.97% | 51.11% |
| Left/right | EEGNet | 46.67% | 46.44% | 51.11% |
| Four-state | CSP + LDA | 53.89% | 41.82% | 50.00% |
| Four-state | CSP + SVM | 48.89% | 28.64% | 50.00% |
| Four-state | EEGNet | 46.67% | 33.13% | 50.00% |

The binary LDA fold accuracies are 80.00%, 53.33%, and 80.00%, showing
substantial recording variability. No tested model achieves the 80% binary or
70% multiclass HLD target. Four-state accuracy is dominated by REST; the SVM
does not exceed the majority-class baseline and has zero FORWARD recall in
this evaluation. These models are not ready for reliable wheelchair navigation.

EEGNet binary fold accuracies are 46.67%, 53.33%, and 40.00%; four-state fold
accuracies are 45.00%, 50.00%, 30.00%, 46.67%, and 63.33%. Pooled four-state
accuracy is weighted by trial count, since the first fold holds out two runs.
The network falls below the majority-class baseline in both tasks. This small
single-subject experiment does not establish that deep learning is generally
worse; it shows that this predefined architecture and training setup do not
generalize adequately here. No hyperparameters were changed after these results.

## Quality And Replay Checks

For S001R04, the selected channels contain 20,000 samples (125 seconds) and
15 REST, 8 left, and 7 right annotations. No selected channel is flat.
Raw channel RMS ranges from 45.20 to 55.75 uV; filtered RMS ranges from
15.55 to 19.14 uV. These statistics do not by themselves prove artifact removal.

Before/after Welch PSD and trace plots were generated and visually inspected.
The plots show attenuation outside 8-30 Hz; separate synthetic tests verify
retention of a 10 Hz signal and suppression of 60 Hz interference.
Local outputs are in `artifacts/quality/S001R04_*`.

A one-second, non-actuating replay smoke test produced four probability
vectors using the real EDF and saved multiclass LDA model. The first inference
took approximately 10.59 ms and later calls approximately 0.65-0.82 ms in that
run. These are illustrative inference-only timings, not a latency benchmark.
The recording is also in the final model's training data, so this replay
provides integration evidence only, not an independent accuracy measurement.

A separate one-second EEGNet replay produced four probability vectors from the
real EDF using the saved four-state checkpoint. It ran without actuation and
does not demonstrate independent accuracy or simulation safety. Its approximately
14.76 ms first call and 1.06-1.26 ms later calls are inference-only smoke-test
timings, not the HLD's complete latency budget. Softmax confidence is uncalibrated.

## Reproduction

From the repository root in Windows PowerShell, after installing requirements:

```powershell
& ./.venv/Scripts/python.exe scripts/download_dataset.py --subjects 1 --runs 4 8 12 6 10 14 --source aws
& ./.venv/Scripts/python.exe -m src.ingestion.preprocessor --subjects 1 --runs 4 8 12 6 10 14
& ./.venv/Scripts/python.exe -m src.ingestion.quality_report --recording data/raw/S001/S001R04.edf
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task binary --classifier lda --output artifacts/binary_lda.joblib
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task binary --classifier svm --output artifacts/binary_svm.joblib
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task multiclass --classifier lda --output artifacts/multiclass_lda.joblib
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task multiclass --classifier svm --output artifacts/multiclass_svm.joblib
```

Each model has a matching JSON report under `artifacts/`. Data and generated
artifacts remain excluded from Git; this summary is suitable for version control.

To reproduce the optional EEGNet comparison on the same processed dataset:

```powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-dl.txt
& ./.venv/Scripts/python.exe -m src.models.trainer --task binary --epochs 50 --batch-size 16 --learning-rate 0.001 --seed 42 --output artifacts/binary_eegnet.pt
& ./.venv/Scripts/python.exe -m src.models.trainer --task multiclass --epochs 50 --batch-size 16 --learning-rate 0.001 --seed 42 --output artifacts/multiclass_eegnet.pt
```

Each `.pt` checkpoint includes CPU model tensors, training-only normalization
statistics, class order, metadata, and the JSON evaluation report. Loading uses
`weights_only=True`. The associated `.json` report records hyperparameters,
per-fold seeds, training loss, and training/validation run assignments.

## Remaining Gates

- Evaluate additional subjects before drawing general conclusions.
- Predefine improvements and tune them using inner training folds, preserving
  outer recording-disjoint evaluation; do not optimize against these scores.
- Validate artifact-rejection criteria, calibration, and predefined model
  improvements using additional independent data rather than tuning on S001 scores.
- Resolve the HLD's five-fold binary protocol versus the three available
  hand-imagery recordings without introducing overlapping-window leakage.
- Validate independent replay, Webots physics, stopping distances, complete
  latency, and memory usage. Webots was not available on the command path.
- Replace offline future-sample filtering before any live-acquisition claim.
- Keep all control simulation-only; no physical wheelchair actuation.