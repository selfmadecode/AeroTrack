# AeroTrack

**Part 1: Person Detection, Tracking, and Counting**

AeroTrack is a computer vision project for detecting and tracking people in drone footage. The first part of the project establishes a baseline using:

- **YOLO** for person detection
- **ByteTrack** for frame-to-frame tracking
- **OpenCV** for video processing and annotation
- **Pandas** for exporting observed tracker IDs

The notebook reads a drone video, assigns temporary IDs to detected people, draws the detections in an output video, and exports the observed IDs to CSV.

> **Counting note:** the project currently counts unique tracker IDs, not proven unique real-world individuals. Occlusion, tracking errors, and a person leaving and re-entering the frame can change the ID or overcount an individual.

## Features

- Detects people with a pretrained YOLO model
- Tracks active detections with ByteTrack
- Assigns temporary tracking IDs
- Annotates the video with bounding boxes and IDs
- Exports an annotated MP4
- Exports a CSV containing observed tracker IDs
- Supports CUDA GPUs and CPU-only execution

## Sample Drone Video

Download a drone video that contains visible people. You can use this [sample video archive](https://www.dropbox.com/scl/fo/9qvpsb3fsamvqzsa12149/AKx_1WK7YqOIf4I0PDX5vRk/Sample.zip?rlkey=7u7131amaul29amyr4jbnnu03&utm_source=chatgpt.com&e=1&dl=0), or download another drone video from a stock-video site or public dataset.

After downloading:

1. Extract the archive if it is a ZIP file.
2. Choose a drone video that shows humans.
3. Rename the video to `drone_video` while keeping its original file extension.
4. Place the renamed video inside `data/`.

Examples:

```text
data/drone_video.mp4
data/drone_video.mov
data/drone_video.avi
```

The file must be a video, not a still image.

## Project Structure

```text
AeroTrack/
├── AeroTrack_Part1_Detection_Tracking_Counting.ipynb
├── data/
│   └── drone_video.mov
├── outputs/
│   ├── drone_tracked.mp4
│   └── observed_tracker_ids.csv
├── yolo26n.pt
└── readme.md
```

| Path | Purpose |
|---|---|
| `AeroTrack_Part1_Detection_Tracking_Counting.ipynb` | Detection, tracking, and counting pipeline |
| `data/drone_video.mov` | Input drone video containing visible people |
| `outputs/drone_tracked.mp4` | Annotated output video |
| `outputs/observed_tracker_ids.csv` | Observed tracker IDs |
| `yolo26n.pt` | Local YOLO model weights |

## Requirements

- Python
- Jupyter Notebook or JupyterLab
- An internet connection for the initial package setup or model download
- A supported drone video

The notebook installs:

- `ultralytics`
- `opencv-python`
- `matplotlib`
- `pandas`

An NVIDIA GPU with CUDA is recommended for faster processing, but the notebook also falls back to CPU.

## Quick Start

1. Clone or download the project.
2. Download any drone video that shows people. You can use a sample video from a stock-video site, public dataset, or any footage you are authorized to use.

   Rename the downloaded video to `drone_video` while keeping its original file extension, and place it in the `data/` folder. For example:

   ```text
   data/drone_video.mp4
   data/drone_video.mov
   data/drone_video.avi
   ```

   Do not change the input filename to an image or use a still image. The notebook requires a video.

3. Open the notebook:

   ```text
   AeroTrack_Part1_Detection_Tracking_Counting.ipynb
   ```

4. Update `VIDEO_PATH` if the video extension or filename differs from `drone_video.mov`, for example:

   ```python
   VIDEO_PATH = DATA_DIR / "drone_video.mp4"
   ```

5. Run every notebook cell from top to bottom.
6. Review `outputs/drone_tracked.mp4`.
7. Review `outputs/observed_tracker_ids.csv`.

For a first test, use a short clip of roughly 20–60 seconds with several visible people and limited motion blur.

## How It Works

1. The notebook detects CUDA availability and selects GPU or CPU execution.
2. The `YOLO` model loads the local weights from `yolo26n.pt`.
3. OpenCV reads the input video one frame at a time.
4. YOLO detects objects, while `classes=[0]` restricts results to the COCO person class.
5. ByteTrack associates detections across frames using `persist=True`.
6. Each active track receives a temporary tracking ID.
7. The notebook draws boxes and labels on every processed frame.
8. Every observed ID is added to a set.
9. The annotated video and sorted unique tracker IDs are exported.

## Configuration

The input and output paths are defined near the start of the notebook:

```python
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

VIDEO_PATH = DATA_DIR / "drone_video.mov"
OUTPUT_VIDEO_PATH = OUTPUT_DIR / "drone_tracked.mp4"
```

To use another video, update `VIDEO_PATH`:

```python
VIDEO_PATH = DATA_DIR / "my_drone_video.mp4"
```

Place the file in the `data/` directory before running the notebook.

## Generated Results

### Annotated video

The processed footage is saved to:

```text
outputs/drone_tracked.mp4
```

The video contains person bounding boxes and tracking labels generated by the YOLO and ByteTrack pipeline.

### Observed tracker IDs

The observed IDs are saved to:

```text
outputs/observed_tracker_ids.csv
```

The CSV has one tracker ID per row:

```csv
tracker_id
1
2
3
```

The final count represents the number of distinct IDs observed during the complete run.

## Example Workflow

The main processing loop:

1. Opens the input video.
2. Processes one frame at a time.
3. Runs person detection and tracking.
4. Records track IDs.
5. Writes the annotated frame to the output video.
6. Displays progress for a subset of frames.

The notebook also provides:

- A detection preview on a selected input frame
- A preview of the completed annotated video
- A sample frame extracted from the annotated output

## Troubleshooting

| Problem | Possible solution |
|---|---|
| Video cannot be opened | Confirm that `VIDEO_PATH` is correct and that the video format is supported by OpenCV |
| Model or package download fails | Check the internet connection and rerun the installation/model-loading cell |
| CUDA is unavailable | Install a compatible NVIDIA driver, CUDA-enabled PyTorch build, and verify that PyTorch sees the GPU; CPU mode will still work |
| Processing is too slow | Use a shorter clip, process fewer frames, or use a CUDA-enabled GPU |
| People are not detected | Use clearer footage with less motion blur and larger or closer visible people |
| Tracking IDs switch unexpectedly | Reduce occlusion or camera movement where possible; this may also require a stronger tracker or future ReID model |
| The output count is too high | Remember that the count is unique tracker IDs, not verified unique people |

## Current Limitations

This is an initial baseline and does not yet provide reliable person re-identification across frames or separate flights.

The current implementation:

- Uses pretrained weights without AeroTrack-specific fine-tuning
- Uses ByteTrack as the tracking baseline
- Uses temporary IDs for one video
- Does not maintain identity across multiple videos or flights
- Does not guarantee an accurate real-world unique-person count
- May lose tracks or create ID switches during occlusion and fast camera motion

## Future Development

Planned improvements include:

- Fine-tuning YOLO on drone-specific annotated footage
- Comparing ByteTrack with BoT-SORT
- Adding appearance-based person re-identification
- Matching identities across multiple video segments and flights
- Improving unique-person counting
- Optimizing processing speed and memory use
- Adding configuration and evaluation scripts

## Status

**Part 1:** initial detection, tracking, and counting baseline implemented.
