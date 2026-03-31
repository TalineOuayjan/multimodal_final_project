# How to Run — Full Pipeline (CSC 52002 Final Project)

**Author:** Taline Ouayjan  
**Course:** CSC 52002 EP 2026 — Multimodal Generative AI

---

## Before You Start — Important Note About Hardware

I want to be upfront about this: the full pipeline (especially the 3D generation step) needs a machine with a dedicated NVIDIA GPU. A standard laptop without a GPU can run Steps 1 and 2 fine, but Step 3 (generating 3D meshes with Hunyuan3D-2) will either fail or take hours on CPU alone.

If you want to see everything working end-to-end, the most reliable option is to run it on the university server (where I developed and tested it), or on any machine with at least a 16 GB VRAM GPU. 

That said, the setup instructions below are complete from scratch — so if you do have access to a GPU machine, everything should just work.

---

## Step 1 — Clone the Repo and Set Up the Environment

You need **Python 3.9** on your machine (`python3 --version` to check). If you don't have it: [python.org/downloads](https://www.python.org/downloads/).

```bash
git clone https://github.com/TalineOuayjan/multimodal_final_project
cd SSV2A

# Create and activate a virtual environment
python3 -m venv ssv2a_env
source ssv2a_env/bin/activate          # Linux / Mac
# ssv2a_env\Scripts\activate           # Windows

# Install all dependencies
pip install -r requirements.txt
pip install gradio fastapi uvicorn soundfile transformers diffusers accelerate openai trimesh
deactivate
```

Then create a **second** environment for Hunyuan3D-2 (3D generation). This one must use **Python 3.10** — the two environments are kept separate because Hunyuan3D-2 has conflicting dependencies with the main pipeline.

```bash
# Check you have Python 3.10 available
python3.10 --version

# Create and set up hy3d_env (must be named exactly hy3d_env, inside the SSV2A folder)
python3.10 -m venv hy3d_env
source hy3d_env/bin/activate           # Linux / Mac

# Install Hunyuan3D-2 and all its dependencies
pip install -r Hunyuan3D-2/requirements.txt
pip install -e Hunyuan3D-2/            # installs the hy3dgen package in editable mode
pip install rembg pymeshlab xatlas numba

deactivate
```

> **Note:** The 3D generation step (Step 3 in the UI) launches `hy3d_env/bin/python` as a subprocess automatically — you do **not** need to activate `hy3d_env` manually when running the pipeline.

---

## Step 2 — Download the Weights

The model weights are too large for GitHub so they're hosted on Google Drive. Download them all from here:

**[Download weights — Google Drive](https://drive.google.com/drive/folders/1ffC2pe2t9qcyQWIThdURfQOynktSxpfW?usp=sharing)**

Once downloaded, place them like this:

```
SSV2A/
├── sam_b.pt                     ← place in the root (next to pipeline.py)
└── weights/
    ├── ssv2a.json               
    ├── dalle2_prior_config.json
    ├── ssv2a.pth                ← from Drive
    ├── agg.pth                  ← from Drive
    ├── dalle2_prior.pth         ← from Drive
    └── yolov8x-oiv7.pt          ← from Drive
```

The following models will download **automatically on first run** from HuggingFace — just make sure you have internet access and about 20 GB of free disk space:
- **Depth-Anything-V2** (~400 MB) — depth estimation
- **Hunyuan3D-2** (~8 GB) — 3D mesh generation, only needed for Step 3
- **AudioLDM / BLIP / CLIP** (a few GB total) — audio generation and captioning

---

## Step 3 — API Key (for the LLM filter)

The pipeline uses GPT-4o Vision in Step 2 to filter out silent objects. The API key is already set up in the `.env` file at the root of the project, so you don't need to do anything.

If for any reason it doesn't work, you can just check **"Skip LLM filter"** in the UI and the pipeline will keep all detected objects without calling the API.

---

## Step 4 — Run the Pipeline

```bash
cd SSV2A
source ssv2a_env/bin/activate
python pipeline.py --port 7875
```

Then open your browser and go to **http://localhost:7875**

---

## Step 5 — Using the Interface

The UI is organized into four sequential steps:

**Step 1 — Detect Objects**  
Upload any image, and click "Detect Objects". YOLO will draw bounding boxes around all detected objects and the depth map will appear on the right. You can uncheck any objects you don't want to include — they'll be grayed out and skipped in all following steps.

**Step 2 — Run Processing Chain**  
Click "Run Processing Chain". This runs the GPT-4o filter (to drop silent objects like walls), SAM segmentation (to get clean cutouts), BLIP captioning (to describe each object in text), and CLIP embedding (to encode each object for audio generation). You can check "Skip LLM filter" if you want to keep all objects.

**Step 3 — Generate 3D World**  
Click "Generate 3D World". This is the slow step — Hunyuan3D-2 generates a 3D mesh for each object (about 2–5 minutes per object on GPU). When done, you'll see an interactive 3D viewer where you can orbit the camera. Use the camera controls to position yourself somewhere in the scene — your position will affect the audio in the next step.

**Step 4 — Generate Spatial Audio**  
Click "Generate Spatial Audio". The pipeline generates a sound for each object (SSV2A/AudioLDM), then mixes them into a stereo output based on your camera position — closer objects are louder, and objects to the left/right pan accordingly. You can move the camera and click "Re-render Audio" to instantly re-mix without regenerating.


## Project Structure (for reference)

```
SSV2A/
├── pipeline.py                      ← main file, run this
├── gradio_3d_world.py               ← standalone 3D + audio (no preprocessing)
├── weights/                         ← all pretrained weights (included)
├── ssv2a/
│   ├── data/
│   │   ├── detect_gemini_SAM.py     ← YOLO detection
│   │   ├── depth_estimation.py      ← Depth-Anything depth estimation
│   │   └── utils.py                 ← CLIP embedding utilities
│   └── model/
│       ├── pipeline.py              ← base SSV2A image-to-audio
│       ├── pipeline_mm2a_SAM.py     ← LLM + SAM + BLIP + CLIP
│       ├── generate_3d_scene.py     ← Hunyuan3D-2 mesh generation
│       ├── spatial_3d.py            ← 3D spatial audio logic
│       ├── aldm.py                  ← AudioLDM wrapper
│       └── clap.py                  ← CLAP embedding
└── Hunyuan3D-2/                     ← 3D generation backend
```
