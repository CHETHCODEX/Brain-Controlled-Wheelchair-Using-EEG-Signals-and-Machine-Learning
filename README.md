# Brain-Controlled Wheelchair Using EEG Signals and Machine Learning

Research prototype based on [the HLD](HLD_for_CapStone.pdf). Recorded EEG is
decoded into navigation intents for a Cyberbotics Webots simulation. This is
not a medical device and must not be connected to a physical wheelchair.

## Team Handoff: Progress As Of 2026-10-20

**Current stage:** working research code, S001 model benchmarks, a local BCI
dashboard, and Webots simulation integration. This is not a completed or
reliable EEG-controlled navigation system. Read this section first when
continuing the project.

### What Has Been Done

| Area | Implemented work | Evidence / limitation |
| --- | --- | --- |
| Dataset | Selective PhysioNet downloader with SHA-256 checks; six S001 imagery runs downloaded locally | Raw recordings are not included in Git; download them again on another machine |
| Preprocessing | Seven-channel selection, notch/bandpass filtering, run-aware labels, 180 epochs, and quality reports/plots | No validated automatic artifact rejection; recorded-data filtering is offline |
| Classical models | Binary and four-state CSP + LDA/SVM, run-disjoint evaluation, saved models and JSON reports | Best S001 accuracy: 71.11% binary, 53.89% four-state; below targets |
| Deep learning | Optional EEGNet training, fold-only normalization, reproducible CPU fitting, and checkpoint replay | S001 accuracy: 46.67% on both tasks; not an improvement over LDA |
| Streaming and transport | Two-second buffer, 250 ms stride, paced EDF replay, and loopback ZeroMQ commands | Replay is not live-headset acquisition or an independent accuracy test |
| Safety controller | Confidence/debounce gates, REST stopping, stale/invalid packet rejection, watchdog, obstacle checks, and caregiver emergency latch | Simulation-only safeguards; not medical safety certification |
| Webots | Wheelchair PROTO, simple arena, furnished hospital rehabilitation scene, surrounding sensors, and telemetry | Existing local native validation report passes a bounded motion/emergency check; full scenarios remain unvalidated |
| Dashboard | Local signal/intent display, command history, simulator telemetry, emergency stop/reset, synthetic input, and external prediction API | Default intents are scripted synthetic inputs, not predictions from a trained EEG model |
| Tests | 39 automated tests passed on 2026-10-20, covering ingestion, quality, models, streaming, transport, controller, adapters, and dashboard | Rerun the suite on your machine; PyTorch-dependent tests may skip without the optional install |

Detailed measured results and evaluation limitations are in
[the S001 benchmark report](docs/experiments/S001_BASELINE.md).
The design reference is [the HLD](HLD_for_CapStone.pdf); the remaining milestone
checklist is in [the project plan](MASTER_PROJECT_PLAN.md).

### Start Here On A Teammate's Machine

```powershell
git clone https://github.com/CHETHCODEX/Brain-Controlled-Wheelchair-Using-EEG-Signals-and-Machine-Learning.git
cd Brain-Controlled-Wheelchair-Using-EEG-Signals-and-Machine-Learning
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
& ./.venv/Scripts/python.exe scripts/generate_hospital_assets.py
& ./.venv/Scripts/python.exe -m src.bci.dashboard --input simulated
```

Open `http://127.0.0.1:8765` (or the URL printed by the server if that port is
busy). The dashboard can run without downloading EEG data or installing
PyTorch. The generated hospital floor texture is used by both the dashboard
and Webots. Stop the dashboard with Ctrl+C before using another publisher.

To connect the simulator, install Webots R2025a or newer, set its Python
command to this checkout's absolute `.venv/Scripts/python.exe` path, and open
[the hospital world](simulation/worlds/hospital_rehab.wbt). Use real-time mode
and check the controller console before running the dashboard. Only one
publisher can bind `127.0.0.1:5555`; do not run the dashboard, scripted
publisher, and EEG replay publisher at the same time. Reload the world when
switching publisher processes to start from a known simulation state.

The existing local `artifacts/hospital-validation.json` records a passing
native scene check: about 0.50 m forward displacement, +0.76/-0.51 rad turns,
0.31 m reverse displacement, and negligible movement under emergency stop.
This generated report is not committed and is not proof of obstacle-course,
watchdog, live EEG, or clinical performance. On Windows with Webots installed
at the launcher's expected per-user location, reproduce the bounded check:

```powershell
& ./.venv/Scripts/python.exe simulation/controllers/scene_validator/scene_validator.py --launch
```

For real-recording experiments, follow the download, preprocessing, and
training commands below. The GitHub checkout does **not** contain `.venv/`,
`data/`, or `artifacts/`: recreate the environment, recordings, processed
epochs, models, quality plots, and runtime state locally. Do not share a
virtual environment or commit raw datasets/checkpoints to make setup work.

### Where To Continue

| Work | Starting files | Next deliverable |
| --- | --- | --- |
| Model evaluation | [classical training](src/models/csp_classifier.py), [EEGNet training](src/models/trainer.py), [benchmark report](docs/experiments/S001_BASELINE.md) | Evaluate S001-S010; report per-subject accuracy, balanced accuracy, and confusion matrices without test-fold tuning |
| Signal quality | [preprocessor](src/ingestion/preprocessor.py), [quality reports](src/ingestion/quality_report.py) | Predefine artifact-rejection criteria using training data and evaluate their effect |
| Simulator validation | [hospital world](simulation/worlds/hospital_rehab.wbt), [native validator](simulation/controllers/scene_validator/scene_validator.py), [controller](src/webots_controller/wheelchair_driver.py) | Record obstacle/turning-clearance, dropout, watchdog, and emergency-reset tests in Webots |
| Dashboard / headset bridge | [dashboard service](src/bci/dashboard.py), [input adapters](src/bci/adapters.py), [dashboard UI](dashboard/app.js) | Connect an independently tested classifier to external mode; preserve fail-closed behavior |
| End-to-end evidence | [EDF replay](src/streaming/inference_engine.py), [tests](tests/test_bci.py) | Run held-out EEG through the complete simulator loop and measure full transition latency and peak memory |

External mode starts with `-m src.bci.dashboard --input external`. It accepts
classified command packets at `POST /api/prediction`, not a raw headset driver.
Packets must satisfy [the command schema](src/ipc/command_schema.py), include
a session ID and increasing sequence IDs, and use same-origin local JSON
requests. Input silence for 0.75 seconds marks the headset disconnected and
returns REST. The trained dataset models have four states; BACKWARD exists
in the simulator/synthetic interface but is **not** a trained EEG class.

**Suggested order:** reproduce setup/tests and the synthetic demo first;
finish simulator failure-case checks next; then improve independent model
evaluation and connect the classifier. A live-headset implementation still
needs its own acquisition integration and validated causal preprocessing.
Keep one focused change per commit, rerun the relevant tests, and update the
benchmark report/project plan with reproducible evidence for completed work.

## Setup (Windows PowerShell)

Use Python 3.12. Run all terminal commands from the repository root.

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
& ./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

If `.venv` already exists, skip its creation. Raw data, cleaned data, and model
artifacts remain local; commit the code and documentation, not those files.

## First EEG Benchmark

Start with S001's three left/right imagery runs rather than the full ZIP:

```powershell
& ./.venv/Scripts/python.exe scripts/download_dataset.py --subjects 1 --runs 4 8 12 --source aws
& ./.venv/Scripts/python.exe -m src.ingestion.preprocessor --subjects 1 --runs 4 8 12
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task binary --output artifacts/binary_lda.joblib
```

The downloader verifies the official SHA-256 checksums, including cached files.
`--source aws` selects the public `physionet-open` S3 mirror listed by PhysioNet;
omit it to use the primary PhysioNet server. The mirror worked during our S001
benchmark when direct downloads stalled.
If PhysioNet is unreachable, retry later or download the EDF files from the
source page into `data/raw/S001/` and rerun the downloader to verify them.
For DNS errors, check the connection, VPN, or DNS settings; do not disable TLS.

Preprocessing selects C3, Cz, C4, FC3, FC4, CP3, and CP4; applies a 60 Hz notch
and 4th-order 8-30 Hz zero-phase Butterworth filter; and extracts 2.5-second
epochs starting one second after each cue. `--notch 50` is also supported.
The model consistently uses the first 320 samples (2 seconds) of each epoch.
This explicit crop reconciles the HLD's offline and streaming window lengths.
No additional automatic artifact-rejection or ICA procedure is implemented.

The CSP + LDA baseline fits CSP within each training fold and holds out entire
recording runs. Three hand-imagery runs permit three recording-disjoint folds,
not the HLD's five folds. The JSON report records accuracy, balanced accuracy,
confusion matrix, and evaluation protocol. `--classifier svm` selects the RBF
SVM alternative. These metrics are not evidence of clinical safety.

## Signal-Quality Inspection

Generate descriptive statistics, before/after Welch PSD plots, and the first
10 seconds of raw/filtered traces for a recording:

```powershell
& ./.venv/Scripts/python.exe -m src.ingestion.quality_report --recording data/raw/S001/S001R04.edf
```

Outputs are written under `artifacts/quality/` as JSON and PNG files. Statistics
include per-channel RMS, peak-to-peak amplitude, mean, flat-channel indices,
and annotation counts. Amplitudes are reported in microvolts. These outputs
support inspection; reduced amplitude alone does not establish clean EEG.
Artifact-rejection thresholds have not been selected or validated.

The first real S001 benchmark achieved 71.1% binary and 53.9% four-state accuracy
with LDA, below the HLD's targets. See [the experiment report](docs/experiments/S001_BASELINE.md)
for SVM and EEGNet results, class imbalance, evaluation protocol, and limitations.

## Four-State Model

Add bilateral imagery runs for REST, TURN_LEFT, TURN_RIGHT, and FORWARD:

```powershell
& ./.venv/Scripts/python.exe scripts/download_dataset.py --subjects 1 --runs 4 8 12 6 10 14 --source aws
& ./.venv/Scripts/python.exe -m src.ingestion.preprocessor --subjects 1 --runs 4 8 12 6 10 14
& ./.venv/Scripts/python.exe -m src.models.csp_classifier --task multiclass --output artifacts/multiclass_lda.joblib
```

T0 maps to REST. For runs 04/08/12, T1 and T2 map to left/right respectively.
For runs 06/10/14, both fists and both feet are deliberately merged into
FORWARD, following the HLD navigation mapping. Movement-execution runs are
rejected. Training is subject-specific; select another participant using
`--subjects` during download/preprocessing and `--subject` during training.

## Optional EEGNet

Install PyTorch and train the compact EEGNet baseline after preparing the same
six-run dataset. Classical training and replay do not require PyTorch; EEGNet
tests are skipped when it is not installed.

```powershell
& ./.venv/Scripts/python.exe -m pip install -r requirements-dl.txt
& ./.venv/Scripts/python.exe -m src.models.trainer --task binary --output artifacts/binary_eegnet.pt
& ./.venv/Scripts/python.exe -m src.models.trainer --task multiclass --output artifacts/multiclass_eegnet.pt
```

The architecture uses temporal convolution, depthwise spatial convolution,
separable convolution, batch normalization, ELU, and spatial dropout. Training
uses 50 fixed epochs, batch size 16, Adam learning rate 0.001, and seed 42 on
one CPU thread. Per-channel normalization is fitted separately within each
training fold. Held-out runs are never used for epoch selection or early stopping.
The final checkpoint is refitted on all task trials and includes normalization,
channel order, label mapping, model weights, and the evaluation report.

S001 EEGNet reached 46.7% binary and 46.7% four-state accuracy, below both LDA
and the respective majority-class baselines. It is an experimental comparator,
not an improved or reliable navigation model. Softmax probabilities have not
been calibrated; a confidence of 0.65 is not a safety guarantee.

Replay the saved four-state checkpoint without actuation:

```powershell
& ./.venv/Scripts/python.exe -m src.streaming.inference_engine --model artifacts/multiclass_eegnet.pt --recording data/raw/S001/S001R04.edf --seconds 10
```

The existing simulator publisher also accepts four-state `.pt` models with
`--publish`, but the same Webots validation gates below still apply. Only load
trusted local checkpoints. EEGNet uses CPU `weights_only=True` loading; classical
`.joblib` files can execute arbitrary code during loading.

## Webots Simulation

1. Install [Cyberbotics Webots](https://cyberbotics.com/#download), R2025a or newer.
2. In Webots preferences, set the Python command to the absolute path of this
    project's `.venv/Scripts/python.exe` so the controller can import `zmq`.
3. Open [the test arena](simulation/worlds/indoor_arena.wbt).
4. Run the simulation in real-time mode, not accelerated mode. The watchdog
    uses wall-clock time to match the external EEG publisher.
5. Verify that the console says `Simulation controller ready` before sending
    commands. In a separate terminal at the repository root, run:

```powershell
& ./.venv/Scripts/python.exe -m src.ipc.zmq_publisher --intent FORWARD --duration 8
```

This is a scripted command test, not EEG decoding. The chair should stop at
the forward obstacle. Reset/reload the world for reproducible starting conditions
before each new publisher process. Session IDs support publisher handover;
retired sessions and duplicate/out-of-order sequence IDs are rejected.
Only one publisher may use port 5555 at a time. Other scripted intents are
REST, TURN_LEFT, and TURN_RIGHT. Commands end after at most 60 seconds.

After training the four-state model, reload the world and replay a recording:

```powershell
& ./.venv/Scripts/python.exe -m src.streaming.inference_engine --model artifacts/multiclass_lda.joblib --recording data/raw/S001/S001R04.edf --publish --seconds 20
```

Omit `--publish` to inspect probabilities without driving the simulation.
Only load trusted local `.joblib` models: this serialization format can execute
code. Binary models cannot publish to the wheelchair because they have no
REST class. Replaying training recordings demonstrates integration only;
use held-out recordings for an independent performance assessment.

## Safety And Limits

EEG replay uses a 2-second buffer with 250 ms stride. Motion requires two
consecutive predictions at confidence >=0.65, a probability margin >=0.15,
and adequate signal quality. REST ramps motion down; emergency stop, low
confidence, malformed commands, invalid sensors, and a one-second heartbeat
timeout brake motion. Obstacle stopping includes a 0.35 m base margin plus a
speed-dependent allowance; forward/reverse obstacle locks require clearance
above 0.6 m and two fresh motion commands to resume. Commands use strict JSON
validation over loopback ZeroMQ.

The wheelchair has two driven wheels and two passive ball-caster approximations.
Front, rear, and side sensors do not guarantee collision avoidance while
turning or cover every obstacle geometry. The bounded hospital motion check
does not validate braking distance, all scene framing, or complete sensor
coverage; these still need scenario-based verification in Webots.

Replay filters the entire EDF recording offline before pacing its samples.
It therefore uses future samples and does not establish causal live-headset
latency. The printed `inference_ms` excludes acquisition, filtering, debounce,
network, and simulator actuation. No end-to-end latency or RAM target has been
certified. Classification thresholds in the HLD are targets, not achieved results.

See [the project plan](MASTER_PROJECT_PLAN.md) for remaining milestones,
including multi-subject evaluation, artifact-rejection criteria, and full
simulation validation.

## Dataset

This project uses the EEG Motor Movement/Imagery Dataset, version 1.0.0,
created by Gerwin Schalk and colleagues at the Wadsworth Center.

- Source: https://physionet.org/content/eegmmidb/1.0.0/
- Official ZIP: https://physionet.org/content/eegmmidb/get-zip/1.0.0/
- Archive size: 1,997,435,063 bytes.
- License: Open Data Commons Attribution License v1.0,
  https://physionet.org/content/eegmmidb/view-license/1.0.0/

The ZIP is excluded from Git history because it exceeds GitHub's ordinary
file-size limit. The selective downloader avoids needing the full archive.

## Attribution

Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0).
PhysioNet. https://doi.org/10.13026/C28G6P

Schalk, G., McFarland, D. J., Hinterberger, T., Birbaumer, N., and Wolpaw,
J. R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI)
System. IEEE Transactions on Biomedical Engineering, 51(6), 1034-1043.
https://pubmed.ncbi.nlm.nih.gov/15188875/

Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical
research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z