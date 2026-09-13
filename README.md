# Plant Leaf Disease Detector

Hybrid CNN + Vision Transformer leaf disease classifier, with YOLO lesion
detection, classical-CV severity scoring, Grad-CAM++ explainability, and
Groq-powered AI advisory reports exported as PDF.

## Project structure

```
.
├── app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
└── models/
    ├── hybrid_model.pth
    ├── lesion_detector_best.pt
    └── class_names.json
```

`models/*.pth` and `models/*.pt` are excluded from git (see **Handling
model weights** below) — everything else should be committed.

## 1. Local setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the secrets template and add your Groq key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit .streamlit/secrets.toml and paste your key
```

Run it:

```bash
streamlit run app.py
```

## 2. Handling model weights

GitHub rejects files over 100 MB, and Streamlit Community Cloud's free
tier has limited resources — so trained checkpoints usually shouldn't be
committed directly. Pick one:

**Option A — external hosting (recommended)**
Upload `hybrid_model.pth`, `lesion_detector_best.pt`, and
`class_names.json` somewhere with a direct-download URL (Hugging Face
Hub, a public S3/GCS bucket, etc.), then add the URLs as secrets:

> **Note:** the training notebook saves the YOLO checkpoint as
> `yolo_runs/lesion_detector_pseudo/weights/best.pt`. Rename/copy it to
> `lesion_detector_best.pt` before uploading — `app.py` looks for that
> exact filename and treats the lesion detector as optional (not an
> error) if it's missing under a different name.

```toml
HYBRID_CKPT_URL = "https://.../hybrid_model.pth"
YOLO_CKPT_URL = "https://.../lesion_detector_best.pt"
CLASS_NAMES_URL = "https://.../class_names.json"
```

`app.py` already checks for these secrets on startup and downloads any
missing file automatically — no code changes needed.

**Option B — Git LFS**
If your files are under GitHub's LFS storage quota:

```bash
git lfs install
git lfs track "models/*.pth" "models/*.pt"
git add .gitattributes
```

Then remove the `models/*.pth` / `models/*.pt` lines from `.gitignore`
before committing.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: plant leaf disease detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Select your repository, the `main` branch, and set the main file path
   to `app.py`.
4. Open **Advanced settings**. In the **"Python version"** dropdown,
   explicitly select **3.11** or **3.12** (see note below — don't skip
   this).
5. In the **Secrets** field, paste the contents of your local
   `.streamlit/secrets.toml` (your Groq key, and the model URL secrets if
   using Option A above).
6. Click **Deploy**. The first build installs PyTorch/Ultralytics and can
   take several minutes.

### Important: `runtime.txt` is currently unreliable

Streamlit Community Cloud has a known, currently-unresolved bug where
`runtime.txt` gets ignored and the platform defaults to a very new Python
version (3.13/3.14 as of late 2026) regardless of what's in the file. This
breaks `torch`, since PyTorch only ships wheels for recent Python
releases — an older pinned `torch` version simply won't have a matching
build. This repo's `runtime.txt` is kept as a hint, but **you must also
select the Python version explicitly in the "Advanced settings" dropdown
when you deploy** — that's the setting that's actually respected.

If you've already deployed and it's stuck on the wrong Python version, you
can't change it in place: delete the app and redeploy, picking the
version in Advanced settings this time.

## Notes

- `requirements.txt` pins CPU-only PyTorch wheels (`+cpu`) — Streamlit
  Cloud has no GPU, and the default CUDA wheels are multiple GB larger,
  which will blow the free tier's build limits.
- `packages.txt` installs `libgl1`/`libglib2.0-0`, which OpenCV and
  Ultralytics need on the minimal Debian image Streamlit Cloud builds on.
- This is a heavy ML stack (Torch + timm + Ultralytics + grad-cam). The
  free Community Cloud tier (1 GB RAM) can be tight — if the app crashes
  on boot, check the app logs for an out-of-memory error first.