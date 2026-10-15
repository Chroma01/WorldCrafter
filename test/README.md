# Camera and prompt guide

Examples are organized under [`test/`](./). Each example includes `prompt.txt`,
`camera.npy`, and `actions.txt`, with an additional `image.png` for I2V.
All examples use the shared `test/negative_prompt.txt` by default.
Run the commands below from the repository root.

## Inference examples

Replace `path/to/` with your input paths. Both modes support `--model-type base`
or `--model-type fast`.

### Image-to-video

```bash
python inference.py --model-type fast --mode i2v \
  --image-path path/to/image.png \
  --prompt path/to/prompt.txt \
  --camera-path path/to/camera.npy
```

### Text-to-video

```bash
python inference.py --model-type fast --mode t2v \
  --prompt path/to/prompt.txt \
  --actions-file path/to/actions.txt
```

### Parameters

Each run saves its video under `output/<timestamp>/`: `i2v_<image-name>.mp4` for I2V,
or `t2v_<prompt-file-name>.mp4` for T2V (`t2v.mp4` for inline text).

| Parameter | Description |
| --- | --- |
| `--model-type` | Select `base` or `fast`. |
| `--mode` | Select `i2v` or `t2v`. |
| `--image-path` | Starting image for I2V; omit for T2V. |
| `--gravity-align` / `--no-gravity-align` | Enable/disable image-based gravity alignment; enabled by default for I2V, not applied to T2V. |
| `--prompt` | Prompt text, a `.txt` file, or an automatic prompt mode for I2V. |
| `--camera-path` | Load a global c2w trajectory from a `.npy` file. |
| `--actions-file` | Generate the trajectory from an action text file. Use `--actions "forward1 yaw_left30"` to pass actions directly instead. |
| `--num-chunks` | Generate only the first N chunks; otherwise use the full trajectory. Each chunk contains 33 frames. |
| `--seed` | Random seed; defaults to `42`. |
| `--enable-compile` | Enable compilation; disabled by default. |

Choose exactly one trajectory input: `--camera-path`, `--actions-file`, or
`--actions`. All three are available in both modes. Negative prompts default to
`test/negative_prompt.txt`; override with `--negative-prompt-path` when needed.
Negative prompts affect Base with CFG enabled; Fast uses CFG=1 and does not use
them for sampling.

## Camera inputs

### Gravity alignment

I2V estimates the input image's full gravity direction with
[GeoCalib](https://github.com/cvg/GeoCalib) before generating the camera path.
Alignment is enabled by default for both Base and Fast. Yaw and horizontal
orbit use the estimated vertical axis; forward/sideways translation follows
the corresponding horizontal plane. The initial image view is preserved,
including its pitch and roll. Vertical orbit retains its camera-local pitch
axis. Alignment does not estimate object depth, choose an orbit target, or
change the generation FOV.

```bash
python inference.py --model-type fast --mode i2v \
  --image-path path/to/image.png --prompt path/to/prompt.txt \
  --actions "forward1 yaw_left30 orbit_left15"

# Preserve an already calibrated or gravity-aligned camera trajectory.
python inference.py --model-type fast --mode i2v \
  --image-path path/to/image.png --prompt path/to/prompt.txt \
  --camera-path path/to/calibrated_camera.npy --no-gravity-align
```

For `--actions` and `--actions-file`, motion is constructed in the estimated
gravity frame. For `--camera-path`, the default assumes an **uncalibrated path
authored for a level initial camera**: relative rotations and translations are
reoriented about the first camera pose. Do not apply this a second time to an
aligned path, or to a real calibrated trajectory; use `--no-gravity-align`.
When `--local-camera-path` is supplied, alignment is automatically bypassed to
preserve the supplied global/local camera pair. Omit that option to derive local
poses from the aligned global trajectory.

Install the updated project dependencies to obtain the pinned GeoCalib package.
Its weights download on first use to the PyTorch cache. Calibration runs on CPU;
results are cached under `$XDG_CACHE_HOME/worldcrafter/gravity` (default
`~/.cache/worldcrafter/gravity`), keyed by model version and preprocessed image.
Use local disk for these caches. Calibration uncertainty is saved as metadata
and does not gate generation. If calibration fails or returns an unusable result,
generation automatically continues with gravity alignment disabled. The fallback
is recorded in calibration metadata without an interface or command-line warning.

The effective camera path and `calibration.json` are saved under
`<video-stem>_trajectory/` next to the output video; the video metadata also
records calibration. Input camera files are not overwritten. T2V does not run
image calibration. For the interactive interface, see the
[Demo guide](../demo/README.md#gravity-alignment).

Define camera motion with actions, or load an existing camera-to-world (c2w)
trajectory. Both options work with I2V and T2V and use 33 frames per chunk.

### Define a trajectory with actions

An action consists of a movement name and a value, optionally followed by `xN`
to repeat it N times. Translation values specify distance in the model's
coordinate scale; rotation values specify degrees. Each action produces one
chunk before repetition. Separate actions with spaces, commas, or newlines;
use `#` for comments.

| Movement | Actions | Short forms |
| --- | --- | --- |
| Forward / backward | `forward1`, `backward1` | `f1`, `b1` |
| Left / right | `left1`, `right1` | `l1`, `r1` |
| Up / down | `up1`, `down1` | Same |
| Turn left / right | `yaw_left30`, `yaw_right30` | `yl30`, `yr30` |
| Look up / down | `pitch_up15`, `pitch_down15` | `pu15`, `pd15` |
| Orbit with a left / right turn | `orbit_left15`, `orbit_right15` | Same |
| Orbit with an upward / downward turn | `orbit_up15`, `orbit_down15` | Same |
| Repeat over N chunks | `forward1xN` | `f1xN` |
| Combine within one chunk | `forward2&right2&yaw_left45` | `f2&r2&yl45` |
| Retrace the preceding N chunks | `reverseN` | Same |

The values above are examples and can be changed. For example, an `actions.txt`
file can contain:

```text
forward1x2
backward1
left1
right1
up1
down1
yaw_left30
yaw_right30
pitch_up15
pitch_down15
orbit_left15
orbit_right15
orbit_up15
orbit_down15
forward2&right2&yaw_left45
reverse2
```

Pass this sequence with `--actions-file path/to/actions.txt`, or use `--actions`
with the actions separated by spaces. Use `--num-chunks N` to generate only the
first N chunks.

Orbit keeps the camera aimed at a point along its current forward direction.
Set the distance to this point with `--orbit-radius 2` (the default is 2), or
put `@orbit_radius 2` at the beginning of the action file. The command-line
value takes precedence. A radius of 0 produces an in-place yaw or pitch turn.
Directions describe the viewing turn: `orbit_left15` turns left while moving
the camera right around the center. Consecutive orbit actions preserve the
center; a translation, in-place turn, or radius change establishes a new one.
Orbit actions occupy separate chunks and support repetition and reversal.
Keep the arc length per chunk, `radius × angle × π / 180`, at most 5.

#### Coordinates and motion

Generated trajectories start at the origin, facing +Z, with +X to the right and
+Y down. Forward/backward and left/right follow the camera's heading on the
horizontal plane, while up/down follows the world vertical axis. Yaw turns the
camera in place, and pitch changes its viewing elevation. Combined translations
follow the heading at the chunk's start, with a maximum translation magnitude
of 5 per chunk. Both the UCPE module and the RepEncoder module derive their
camera representations from the same global c2w trajectory: the UCPE module
uses chunk-relative poses at metric scale, while the RepEncoder module
normalizes translations by a scene scale computed from the selected source views.

### Export or load camera poses

Inference converts actions to c2w poses automatically. To export a trajectory
without loading a model, run:

```bash
python tools/build_trajectory.py \
  --actions-file path/to/actions.txt \
  --output-dir output/trajectory
```

Use the resulting file with `--camera-path output/trajectory/camera.npy` in place
of `--actions-file` or `--actions`.

You can also supply your own `.npy` file containing global c2w matrices with
shape `[T, 3, 4]` or `[T, 4, 4]`. Use the same right/down/forward convention,
with one pose per frame and 33 frames per chunk. Follow the coordinate and
scale conventions described above.

## Prompt styles

### Third-person following views

<img src="I2V/00_cat_robot_vacuum/image.png" alt="Cat input image" width="640">

For a moving subject that should stay in view, begin with
**“A third-person ... view closely follows ...”** to encourage subject following.
Describe the subject's appearance, its movement, and its surroundings.
For example, the [Cat prompt](I2V/00_cat_robot_vacuum/prompt.txt) starts:

> A third-person gameplay-like camera closely follows a gray robot vacuum moving through a modern interior with reflective hardwood floors and beautiful rays of light.

The rest of the prompt describes the cat, the furniture, and how the cat balances
on the moving vacuum.

### First-person views

<img src="I2V/03_waterfall/image.png" alt="Waterfall input image" width="640">

For first-person views, describe the setting, spatial layout, materials,
lighting, and relationships between objects. For example, the
[Waterfall prompt](I2V/03_waterfall/prompt.txt) starts:

> A broad garden waterfall pours over layered dark rocks into a shallow pool surrounded by dense subtropical plants.

It then describes the rock formations, surrounding foliage, and pool boundaries
to establish the scene's structure.

<img src="I2V/01_socrates/image.png" alt="Socrates input image" width="640">

The [Socrates prompt](I2V/01_socrates/prompt.txt) starts:

> A scene of static, painted sculptures depicts a solemn stone prison chamber, with classical figures neatly arranged around a low wooden bed.

It describes the figures as painted sculptures, then details their poses,
clothing, props, and arrangement within the chamber.

### Length and consistency

Use one focused English paragraph. Around **80–120 words** is a useful starting
point; dynamic subject-following prompts often need **100–130 words** to cover
both motion and environment. These are writing guidelines, not input limits.

For I2V, keep the description consistent with the input image. For T2V, describe
the subject and setting explicitly because there is no starting image. Keep
appearance and lighting consistent throughout the paragraph, and avoid cuts,
shot changes, or camera directions that compete with the supplied trajectory.
The bundled prompts preserve the wording used for their original examples.

### Automatic prompts

Generate a prompt from your input image with either model:

```bash
python inference.py --model-type fast --mode i2v \
  --image-path path/to/image.png \
  --prompt auto-first-person \
  --actions-file path/to/actions.txt
```

Use `auto-first-person` for a description of the scene's layout, materials, and
lighting, or `auto-third-person` for a subject-following description beginning
with “A third-person ... view closely follows ...”. Both templates target one
English paragraph of 100–130 words. A single image leaves future motion ambiguous;
review the generated prompt when a particular action matters.

The script loads `Qwen/Qwen3-VL-4B-Instruct` on the selected `--device`, generates
the prompt, and releases the caption model before loading WorldCrafter. Its weights
are loaded from `weights/Qwen3-VL-4B-Instruct` when present, otherwise downloaded
and cached on first use. Use `--caption-model path/to/Qwen3-VL-4B-Instruct`
to select another location. The [first-person](../worldcrafter/prompts/first_person.txt)
and [third-person](../worldcrafter/prompts/third_person.txt) templates can be edited
to adjust the writing style.

The resolved text and caption metadata are saved beside the video as
`<video-stem>.prompt.txt` and `<video-stem>.caption.json`. Reuse the text with
`--prompt path/to/video.prompt.txt` to reproduce or edit it. Values ending in
`.txt` are interpreted as file paths; a missing or empty file raises an error.
Base runs with `--state-output-dir` also
save `auto_prompt.json` alongside the states; resuming with the same auto mode
reuses this prompt. Automatic prompts require I2V input; ordinary text prompts
and prompt files use the existing inference path.
