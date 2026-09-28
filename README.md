# NetraSetu

NetraSetu is a hackathon prototype for rural diabetic-retinopathy screening. A village health worker registers a patient, records structured history, uploads fundus images for both eyes, receives an image-quality verdict and creates a prioritized case. An ophthalmologist reviews the original images, confirms or corrects the preliminary grade, returns instructions and generates a PDF report.

> Research and demonstration software only. It is not a validated medical device and must not be used for autonomous diagnosis or treatment.

## What is included

- Village health-centre dashboard
- Guided patient registration and consent
- Right- and left-eye image upload
- Explainable blur, exposure, glare and field-coverage checks
- Five-grade PyTorch inference interface
- Safe APTOS-label demonstration mode when no checkpoint is installed
- Transparent urgency rules
- Prioritized ophthalmologist queue
- Confirm, correct, recapture and referral actions
- Doctor-reviewed PDF report
- SQLite persistence and audit events
- APTOS EfficientNet-B0 training script
- Unit tests for safety-critical triage rules

## Run locally

On Windows, double-click `setup.bat` once. After it completes, double-click
`run.bat` whenever you want to start the application.

Alternatively, from PowerShell in this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print the local browser address, normally `http://localhost:8501`.

## Demonstration flow

1. Open the **Village health centre** workspace.
2. Click **Load three demonstration cases**.
3. Inspect the routine, referral and urgent cases.
4. Switch to the **Ophthalmologist** workspace.
5. Select the urgent case, review both images and sign a decision.
6. Return to the village workspace and download the reviewed report.

The demonstration cases are synthetic patient records paired with APTOS images. Until `models/dr_model.pth` exists, the app explicitly uses the dataset's reference labels and shows a warning. It never disguises reference labels as model predictions.

## Train the baseline model

Expected data layout:

```text
data/raw/aptos/
├── train.csv
└── train_images/
```

Train with:

```powershell
python train.py --epochs 5 --image-size 384
```

The best validation checkpoint is saved to `models/dr_model.pth`. Training is substantially faster on a Kaggle or Colab GPU. The baseline uses a stratified image split because APTOS does not publish reliable patient identifiers; this limitation must be disclosed and a patient-grouped local validation set should be used before any real pilot.

## Project structure

```text
app.py                 Streamlit application
train.py               APTOS baseline training
src/database.py        SQLite records and audit trail
src/imaging.py         Image-quality assessment
src/inference.py       Model inference and safe demo fallback
src/triage.py          Versioned urgency rules
src/report.py          Doctor-reviewed PDF generation
tests/                 Unit tests
```

## Intentionally outside the hackathon MVP

- Production authentication
- Real ABHA/ABDM integration
- Autonomous diagnosis
- Electronic prescribing integration
- Lesion segmentation
- Offline synchronization
- Regulatory or clinical validation
