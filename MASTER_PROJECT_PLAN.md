# Implementation And Verification Plan

## Current Milestone: Model Baselines, Dashboard, And Simulation Integration

Handoff updated 2026-10-20. The full automated suite passes 39 tests.
See [the README handoff](README.md) for teammate setup and continuation tasks.

- [x] Run-aware labels and selective checksum-verified EDF downloader.
- [x] Channel selection, zero-phase filtering, epoching, and synthetic tests.
- [x] CSP + LDA/SVM training with run-disjoint evaluation and saved reports.
- [x] Two-second ring buffer and offline EDF replay at 250 ms stride.
- [x] Local ZeroMQ command protocol and real loopback transport test.
- [x] Debounce, low-confidence stopping, timestamp/sequence validation, watchdog,
      obstacle stopping, and pure-Python controller tests.
- [x] Native Webots wheelchair PROTO, test arena, and Python adapter files.
- [x] Download and checksum-verify all six real S001 imagery recordings through
      the official public AWS mirror after direct-server DNS/stalling problems.
- [x] Preprocess 180 epochs and run binary/multiclass LDA and SVM benchmarks.
      See [the S001 experiment report](docs/experiments/S001_BASELINE.md).
- [x] Smoke-test real EDF replay and saved-model probabilities without actuation.
- [x] Add a hospital rehabilitation world, front/rear/side sensors, reverse
      simulation commands, motion ramps, and obstacle-clearance locks.
- [x] Add the local BCI dashboard with synthetic input, external-classifier
      interface, simulator telemetry, and a persistent caregiver emergency latch.
- [x] Record a passing bounded native hospital scene check for stationary
      sensors/pose, forward/reverse movement, turn signs, and emergency stopping.
      Local evidence is in ignored `artifacts/hospital-validation.json`;
      reproduce it using the scene validator before relying on it elsewhere.
- [ ] Open the world in Webots and verify loading, framing, motor directions,
      passive caster stability, sensor readings, and collision geometry across
      scenarios beyond the bounded hospital check, including the simple arena.
- [ ] Verify forward/left/right/rest using scripted commands in real-time mode.
- [ ] Confirm obstacle stops and <=1-second watchdog behavior in Webots itself.
- [ ] Replay independently held-out EEG through the complete simulator loop.

## Subsequent HLD Milestones

- [x] Add before/after PSD, signal traces, descriptive quality statistics, and
      synthetic spectral tests; generate and inspect real S001R04 plots.
- [ ] Define artifact-rejection criteria using training data rather than
      arbitrary preprocessing choices, then evaluate without tuning on test data.
- [ ] Improve and independently evaluate decoding: S001 LDA achieved 71.1%
      binary and 53.9% four-state accuracy, below the 80%/70% HLD targets.
- [x] Add PyTorch EEGNet, training-only normalization, run-aware validation,
      reproducible CPU seeds, checkpoint/replay support, and S001 comparison.
      Fixed 50-epoch EEGNet achieved 46.7% accuracy on both tasks, below LDA
      and the respective majority baselines; accuracy goals remain unmet.
- [ ] Validate prediction calibration and continuous-stream performance before
      interpreting the HLD's confidence threshold as meaningful reliability.
- [ ] Benchmark subjects S001-S010 and report per-subject and aggregate results.
- [ ] Resolve the five-fold evaluation requirement: three hand-imagery runs
      support only three recording-disjoint folds. Do not silently substitute
      shuffled or overlapping windows and label the results equivalent.
- [ ] Measure full command-transition latency, including debounce, transport,
      and actuator steps, plus peak inference memory. Do not equate inference
      timing alone with the HLD's end-to-end latency target.
- [ ] Build richer indoor scenarios and validate stopping distance, turning
      clearance, occlusion, controller disconnection, and stale-message recovery.
- [ ] If live acquisition is added, replace offline future-sample filtering
      with a stateful causal pipeline and retrain/validate under that pipeline.
- [ ] Produce experiment reports, test evidence, and the final capstone demo.

## Explicit Design Decisions

- Data and artifacts stay local; source code, tests, and documentation go to Git.
- The repository uses `data/raw` and `data/processed`; the HLD's `DATASET`
  directory name is not required by the processing interface.
- Preserve 2.5-second offline epochs, but train and infer on the same 2-second
  crop. Zero-phase replay is an offline demonstration, not live BCI validation.
- Merge bilateral fists/feet imagery into FORWARD. Labels always include run
  context so unilateral and bilateral T1/T2 annotations cannot be confused.
- Low confidence stops motion instead of retaining an earlier moving intent.
- REST ramps motion down; faults and emergency stops brake immediately.
      Debounce transitions into moving states.
- Debouncing is enforced in the simulator controller, not trusted to a sender.
- Publisher sessions support handover while rejecting retired sessions and
      duplicate/out-of-order packets; reload worlds for reproducible test starts.
- The hospital world has a passing local bounded native motion check, not
      complete obstacle/watchdog/EEG-loop validation or clinical safety evidence.
- Synthetic dashboard intents are scripted, not trained-model predictions.
      BACKWARD is supported by the simulator, not the four-state EEG classifiers.
- PyTorch is optional via requirements-dl.txt. EEGNet uses fixed-epoch,
  training-fold-normalized CPU fitting; held-out runs do not select epochs.
- Simulation only. No physical wheelchair control or medical safety claim.