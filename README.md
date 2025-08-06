# Eye Tracker Project

The eye tracking project that landed me my first data science job. It places the company logo over detected eyes (handles multiple people).

The architecture consists of a `VideoStreamPipeline` that can take in a `computer`, which is simply any computations that should be done on each frame. A computer is expected to take in a frame, modify it in any way and return the modified frame.

I used my Python-based [cookiecutter repository](https://github.com/JeffreyWardman/cookiecutter-python) to create the baseline repository. This project is an example of how I like to structure my codebases (minus unit/integration tests).

## Installation

The below will create a virtual environment and install the required dependencies.

`make install`

## Usage

`.venv/bin/python -m eye_tracker.main`

or

`docker compose up`

## Limitations

- Bounding rectangle is not rotated, which has the trade-off of better accuracy of overlaying on the eyes for compute cost.
- `dlib`'s shape predictor model is slow, inefficient and computed on the CPU. There are significant ways to improve the accuracy and inference time of this step.
- No GPU acceleration for rendering via OpenCV.
- Docker image does not work for macOS as direct camera passthrough is not supported. Did not extend this project to stream the camera output over the network.

## Omitted

- Unit tests
