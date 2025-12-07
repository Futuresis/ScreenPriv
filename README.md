## GUI Agent Privacy Detection VLM Tool

This project is a lightweight utility script that leverages a Vision-Language Model (VLM) to detect privacy-related information **specifically in screenshots captured during the execution of GUI (Graphical User Interface) agents**, and it exports both annotated images and a JSON label file as the output.

**A GUI Agent (Graphical User Interface Agent)** is an artificial intelligence agent based on Visual Language Models (VLMs) that perceives its environment exclusively through graphical user interfaces (e.g., screen screenshots) and simulates human operations to complete cross-platform tasks.

> For a Simplified Chinese version of this document, see: **[README_zh.md](README_zh.md)**.

### 1. Environment & Installation

- **Python version**: Python 3.8+ is recommended.
- **Required pip packages** (minimum):

```bash
pip install openai pillow
```

### 2. Configure `base_url` and `api_key`

The OpenAI-compatible client is created near the top of `privacy.py`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="",
    api_key=os.getenv("OPENAI_API_KEY", ""),
)
```

You have two options:

- **Option A – Use environment variable (recommended)**  
  Set the environment variable before running the script:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

- **Option B – Hard-code the key (for quick local testing only)**  
  Edit `privacy.py` directly:

```python
client = OpenAI(
    base_url="",
    api_key="your_api_key_here",
)
```

### 3. Data Layout

The script expects each task to be stored in a folder with the following structure:

```text
<task_folder>/
  task_result.json      # contains "goal"
  traj.jsonl            # each line has a "manager" object with "response"
  images/
    *.png               # screenshots for each step
```

Example (from this repo):

```text
data/Gmail_View_third_email_in_Sen/
  task_result.json
  traj.jsonl
  images/
    screenshot_....png
```

### 4. How to Run

Basic command:

```bash
python privacy.py <task_folder> --model <model_name>
```

```bash
python privacy.py data/Gmail_View_third_email_in_Sen --model "google/gemini-3-pro-preview"
```

What the script does:

1. Reads the **goal** from `task_result.json`.
2. Reads the sequence of **manager responses** from `traj.jsonl`.
3. Pairs each response with the corresponding screenshot under `images/`.
4. Builds a detailed privacy-classification prompt (see `get_prompt_template()`).
5. Calls the VLM with both text prompt and image.
6. Parses the model output and:
   - Draws bounding boxes on the original images under  
     `annotations/<model_name>/..._annotated.png`
   - Writes a consolidated JSON file:  
     `annotations/<model_name>/ai_results.json`

### 5. About Claude and Coordinate Normalization

Different VLM providers may use different **coordinate systems** or normalization conventions in their responses.  
The function `convert_normalized_coords_to_pixels()` in `privacy.py` currently assumes a simple **0–1000 normalized coordinate system**:

```python
x1 = coords["x1"] * width / 1000
y1 = coords["y1"] * height / 1000
x2 = coords["x2"] * width / 1000
y2 = coords["y2"] * height / 1000
```

For some Claude setups, coordinates may effectively be normalized over a **fixed reference resolution**, e.g. for 1080x2400 images you might empirically find:

- \( \text{scale\_x} = \text{img\_width} / 705 \)
- \( \text{scale\_y} = \text{img\_height} / 1567 \)

In that case, you would adjust the conversion accordingly, for example:

```python
scale_x = img_width / 705
scale_y = img_height / 1567
x1 = coords["x1"] * scale_x
y1 = coords["y1"] * scale_y
...
```

Because this is **provider- and model-specific**, you should:

1. Run the script on a few **test screenshots**.
2. Inspect the annotated images under `annotations/<model_name>/`.
3. Manually tune the normalization formula in `convert_normalized_coords_to_pixels()`  
   until the drawn boxes align well with the objects described by the model.

### 6. Notes and Limitations

- The script expects the model to follow the exact output format described in the prompt template.
- Lines that do not parse correctly are skipped silently in `parse_ai_output()`.
- This is a light-weight research / demo script; it does **not** include robust error handling, rate limiting, or batching.
- The quality of the model output may be unsatisfactory for two main reasons:  
  (1) the model’s understanding of the concept of “privacy” is limited, so it may fail to detect some privacy information in the image;  
  (2) the model’s grounding capability is limited, so even if it recognizes privacy-related content, it may not be able to provide sufficiently accurate locations (coordinates).

