### Det-SAM2-pipeline

Our tech report in https://arxiv.org/abs/2411.18977

The Det-SAM2 project is a pipeline based on the Segment Anything Model 2 segmentation model ([SAM2](https://github.com/facebookresearch/sam2)) that uses the YOLOv8 detection model to automatically generate prompts for SAM2. It further processes SAM2's segmentation results through scenario-specific business logic, achieving fully automated object tracking in videos without human intervention. This implementation is tailored for billiard table scenarios.

For the SAM2-compatible predictor, our core contributions include:

- Development of a **self-prompted video instance segmentation pipeline** (Det-SAM2-pipeline) that requires no manual interaction. It supports inference and segmentation of specific categories (determined by a custom detection model) from **video streams** and returns segmentation results with SAM2's original precision to enable further business-level processing.
- We implemented functionality to **add new categories during inference and tracking** without interrupting the inference process.
- Our pipeline allows applying the memory bank from one video inference session to new videos. We call this a **preload memory bank**, enabling the pipeline to leverage the inferred memory (object categories, shapes, and motion states) from a previous video for inference on new videos **without requiring additional prompts for the new video**.
- We achieved **constant GPU and memory usage** in the Det-SAM2-pipeline, enabling inference for videos of unlimited length.

If you are optimizing SAM2 in an engineering context, we highly encourage you to refer to the implementation of Det-SAM2. Additionally, we built a complete pipeline using Det-SAM2 that handles business scenarios (billiard table) such as shot recognition, ball collision, and boundary rebound detection. Previously, traditional non-SAM2 tracking algorithms struggled to accurately address these three conditions in fast-moving billiard table scenarios.

**Note:** Our open-source scripts are annotated entirely in **Chinese** to facilitate development, without affecting functionality. If needed, you can use tools like ChatGPT to translate the annotations into your preferred language when referencing the script's functionality.



### Installation

------

Our project is built based on version 2.1 of SAM2. The environment dependencies are almost identical, so you can deploy it by following the installation instructions for SAM2.1: https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation.

In addition, there may be a few extra packages that need to be installed separately. Please refer to the error messages and install them as required (these are common packages and shouldn't be many).

Alternatively, you can directly access the image we have published on AutoDL at https://www.codewithgpu.com/i/motern88/Det-SAM2/Det-SAM2

Most of the executable files in our project are located in the `det_sam2_inference` folder under the root directory of `segment-anything-2`.

```python
segment-anything-2/det_sam2_inference:
├──data
│   ├──Det-SAM2-Evaluation
│   │   ├──videos
│   │   ├──postprocess.jsonl  # annotation
│   ├──preload_memory_10frames  # read frames to build preload memory bank

├──det_weights
│   ├──train_referee12_960.pt  # yolov8n, our example weight in billiards scenario

├──eval_output/eval_result
│   ├──eval_results.json
│   ├──result_visualize.py  # visualize eval_results.json(eval_det-sam2.py output)

├──output_inference_state
│   ├──inference_state.pkl  # generated preload memory bank

├──pipeline_output
├──temp_output
│   ├──det_sam2_RT_output  # det_sam2_RT.py visualize output
│   ├──prompt_results  # SAM2 prompt (by detect model) visualize output 
│   ├──video_frames

Det_SAM2_pipeline.py  # Det-SAM2 + post-process pipeline
det_sam2_RT.py  # Det-SAM2 process function
eval_det-sam2.py  # find optimal parameter combination
frames2video.py
postprocess_det_sam2.py  # post-processing example (billiards scenario)
```

Additionally, Det-SAM2 introduces the following modifications compared to SAM2 (our changes only add new features without removing any of the official functionalities implemented in SAM2.1):

```python
segment-anything-2/sam2:
├──modeling
│   ├──sam2_base.py
├──utils
│   ├──misc.py
sam2_video_predictor.py
```

#### Checkpoints

We use the SAM2.1 weights:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

For the detection model, you can use any YOLOv8 weights of your choice or start with our billiard detection model trained specifically for billiard scenarios:

- [Det-SAM2-YOLO8-Weight](https://huggingface.co/ATA-space/Det-SAM2/tree/main/det_weights)



### Getting Started

------

The scripts below demonstrate post-processing judgment in a billiard scenario by default. If you only need to use the Det-SAM2 framework, simply run `det_sam2_RT`.



**1**.Execute Det-SAM2 segmentation mask prediction and post-processing scripts separately:

Use the detection model to automatically provide prompts for SAM2, which then performs segmentation predictions on the video:

```cmd
python det_sam2_inference/det_sam2_RT.py
```

The `det_sam2_RT.py` script defines the `VideoProcessor` class (**parameter settings are explained below**), with its primary method being `VideoProcessor.run()`.

After running `det_sam2_RT.py` and saving the segmentation result dictionary `self.video_segments`, use the post-processing script for business logic analysis on the segmented masks:

```cmd
python det_sam2_inference/postprocess_det_sam2.py
```

The `postprocess_det_sam2.py` script defines the `VideoPostProcessor` class (**parameter settings are explained below**), with its primary method being `VideoPostProcessor.run()`.



**2**.Run the end-to-end pipeline script (supports constant GPU and memory usage), enabling asynchronous parallel inference of segmentation masks and post-processing judgments.

Execute the full pipeline script to infer long videos in one go:

```cmd
python det_sam2_inference/Det_SAM2_pipeline.py
```

The `Det_SAM2_pipeline.py` script defines the `DetSAM2Pipeline` class (**parameter settings explained below**). This class processes real-time video stream inference asynchronously and in parallel, combining the segmentation backbone (`VideoProcessor`) and post-processing (`VideoPostProcessor`) functionalities. Its primary method is `DetSAM2Pipeline.inference()`.



**3**.Run the automated evaluation script to explore various parameter combinations:

```cmd
python det_sam2_inference/eval_det-sam2.py
```

The `eval_det-sam2.py` script defines the `EvalDetSAM2PostProcess` class (**parameter settings explained below**). This class loops through multiple candidate parameter combinations to infer the entire evaluation dataset. For each sample inference, the segmentation backbone (`VideoProcessor.run()`) and post-processing (`VideoPostProcessor.run()`) are executed sequentially. The evaluation results are collected and written to `eval_results.json`. The primary method for this script is `EvalDetSAM2PostProcess.eval_all_settings()`.

After completing the `eval_results.json` file, you can visualize the evaluation results for different parameter settings:

```cmd
python det_sam2_inference/eval_output/eval_result/result_visualize.py
```



#### Parameter Parsing

------

Below is an explanation of the parameters for the key classes and their main functions in the scripts `det_sam2_RT.py`, `postprocess_det_sam2.py`, `Det_SAM2_pipeline.py`, and `eval_det-sam2.py`.

- **Parameters for initializing the `VideoProcessor` class** in `det_sam2_RT.py`:
  - **`output_dir`**: Path to save rendered results. Default: `"./temp_output/det_sam2_RT_output"`.
  - **`sam2_checkpoint`**: Path to the SAM2.1 model checkpoint, e.g., `"../checkpoints/sam2.1_hiera_large.pt"`.
  - **`detect_model_weights`**: Weights of your YOLOv8 detection model trained for specific classes. In the billiards scenario example, use `"../checkpoints/sam2.1_hiera_large.pt"`.
  - **`detect_confidence`**: Confidence threshold for the YOLO detection model. In the example, this is `0.85`.
  - **`skip_classes`**: IDs of YOLO detection model classes you wish to ignore as prompts for SAM2. For instance, if you want to skip classes 11, 14, 15, and 19, set `skip_classes={11, 14, 15, 19}`.
  - **`vis_frame_stride`**: Interval for rendering SAM2 segmentation results. Set to `-1` to disable rendering. Default: `-1`.
  - **`visualize_prompt`**: Whether to visualize interaction/condition frames (frames with detection prompts). Useful for verifying if detection prompts are correct. Default: `False`.
  - **`frame_buffer_size`**: Determines the number of video frames accumulated before SAM2 inference. For efficiency, inference is not performed for every frame. By default, the buffer processes 30 frames at a time, i.e., `frame_buffer_size=30`.
  - **`detect_interval`**: Controls the interval at which the detection model adds prompts to SAM2. The default is `30`, meaning detection occurs every 30 frames. Set to `-1` to disable detection, but there must be at least one condition frame for inference. If using a preloaded memory bank (with all frames as condition frames), this can be set to `-1`. This parameter determines the frequency of condition frames.
  - **`max_frame_num_to_track`**: Limits the inference propagation length during SAM2 video inference (`SAM2VideoPredictor.propagate_in_video`). Past frames beyond this length are considered to have sufficient information and are not re-inferred. Default: `60`. Must be at least twice `frame_buffer_size` to ensure all frames can be corrected in subsequent propagations.
  - **`max_inference_state_frames`**: Part of memory optimization. Retains only a limited number of memory bank frames. Frames exceeding this limit are released. Default: `60`. Must be greater than or equal to `max_frame_num_to_track`. Increase this parameter if memory allows, as it retains more useful computation data.
  - **`load_inference_state_path`**: Path to load a preloaded memory bank (in `.pkl` format). Default: `None`.
  - **`save_inference_state_path`**: Path to save a memory bank for inference transfer to new videos. Default: `None`. For example, set this to `"output_inference_state/inference_state_frames.pkl"`. Ensure `max_inference_state_frames` is sufficiently large to provide enough valid memory information.



- **Parameters for the `VideoProcessor.run()` inference function**:
  - **`video_path`**: Path to the input MP4 video (optional, mutually exclusive with `frame_dir`). Default: `None`.
  - **`frame_dir`**: Path to a directory containing video frames (optional, mutually exclusive with `video_path`). Default: `None`.
  - **`output_video_segments_pkl_path`**: Path to save the dictionary of segmentation masks. This dictionary collects SAM2's output masks for post-processing or other operations. Default: `"./temp_output/video_segments.pkl"`.
  - **`output_special_classes_detection_pkl_path`**: Path to save special class detection results in a `.pkl` file. If detection results are needed without SAM2 segmentation, this is where the results are saved. Default: `"./temp_output/special_classes_detection.pkl"`.



- **The `VideoPostProcessor` class in `postprocess_det_sam2.py`**： is a post-processing class for the billiard scene based on Det-SAM2 output segmentation results. It is used to determine whether a ball goes into the pocket, whether balls collide, and whether a ball rebounds from the table edge. The initialization parameters of this class are strongly related to the video input resolution. The default parameters below are based on a resolution of 1920*1080. Specific scene samples can be found in the evaluation set at [Det-SAM2/data/Det-SAM2-Evaluation](https://huggingface.co/ATA-space/Det-SAM2/tree/main/data/Det-SAM2-Evaluation):

  - `pot_distance_threshold`: Default is `100`. This threshold is used to determine whether the ball is near the pocket. Increasing this value will provide a larger detection range.
  - `pot_velocity_threshold`: Default is `0.9`. This threshold is used to determine the direction of the ball’s velocity vector when it enters the pocket. Increasing this value allows for more lenient deviation from the pocket’s direction.
  - `ball_distance_threshold`: Default is `120`. This is the threshold for determining if balls are close enough to collide. The collision is considered only if the distance between the two balls is within this value.
  - `ball_velocity_threshold`: Default is `10`. This threshold is used to determine if a ball collision occurred based on the change in velocity (acceleration) after the collision. If the change exceeds this threshold, the collision is considered to have happened.
  - `table_margin`: Default is `100`. This value creates a buffer zone extending a certain length from the table's edge to account for potential rebounds. Increasing this value will expand the area where a ball may rebound from the table edge.
  - `rebound_velocity_threshold`: Default is `0.7`. This threshold is used to detect if a ball has rebounded from the table edge. Increasing this value makes it easier for the ball to be detected as having rebounded (indicating that the change in vertical velocity components before and after the collision does not exceed this threshold).

  

- **The `VideoPostProcessor.run()` inference function requires the following parameters:**

  - `segments_dict_pkl`: Path to the PKL file containing the SAM2 segmentation dictionary, automatically generated by `det_sam2_RT.py`.
  - `time_interval`: Default is `1.0`. This is the time interval used to calculate the velocity vector. Typically, the velocity vector is calculated based on the interval between consecutive frames.

  

- **The `DetSAM2Pipeline` class in `Det_SAM2_pipeline.py`** : initializes the `VideoProcessor` and `VideoPostProcessor` classes and asynchronously processes SAM2 inference and post-processing functions in the `DetSAM2Pipeline.inference()` method.

  - `sam2_output_frame_dir`: Directory to save the SAM2 segmentation visualization results. Default is `"./temp_output/video_frames"`.
  - `sam2_checkpoint_path`: Path to the SAM2 model weights. Default is `"../checkpoints/sam2.1_hiera_large.pt"`.
  - `sam2_config_path`: Path to the SAM2 configuration file. Default is `"configs/sam2.1/sam2.1_hiera_l.yaml"`.
  - `detect_model_weights`: Path to the YOLOv8 detection model weights. In the billiard scene example, the path is `"det_weights/train_referee12_960.pt"`.
  - `output_video_dir`: Directory for the post-processed visualization output. Default is `"./pipeline_output"`.
  - `load_inference_state_path`: Preloaded memory bank. Default is `None`. If a preloaded memory bank is needed, provide the path to it (the preloaded memory bank is automatically generated by the `det_sam2_RT.py` script, as explained in its parameter section).
  - `visualize_postprocessor`: Whether to visualize post-processing results. Default is `False`. If post-processing results are visualized, constant memory overhead cannot be used, and the `VideoProcessor` class must retain all information in `video_processor.inference_state["images"]` to support post-processing visualization. Specifically, this is achieved by passing a `max_inference_state_frames` value larger than the total number of frames in the current video during the initialization of `VideoProcessor` in `DetSAM2Pipeline.__init__()`.

  Note: During the initialization of the `DetSAM2Pipeline.__init__()` class, the `VideoProcessor` and `VideoPostProcessor` classes are instantiated. The `VideoProcessor` is reinitialized with parameters passed during initialization, while the `VideoPostProcessor` uses the default settings from `postprocess_det_sam2.py`.



- **The `DetSAM2Pipeline.inference()` inference function requires the following parameters:**

  - `video_source`: The video source, which can either be a local MP4 video path or an RTSP URL.
  - `max_frames`: The maximum number of frames to process from the video stream. Once this frame limit is reached, the inference will end manually. The default is `2000`. In theory, it can handle an unlimited number of frames. `DetSAM2Pipeline.inference()` is implemented to maintain constant GPU/CPU memory usage.

  

- **The `EvalDetSAM2PostProcess` class in `eval_det-sam2.py`**: is used to find the optimal initialization parameter combination for `VideoProcessor` and `VideoPostProcessor` in the evaluation dataset. Its initialization parameters are:

  - `sam2_output_frame_dir`: Temporary folder for storing SAM2 output mask frames. The default is `"./temp_output/det_sam2_RT_output"`.
  - `sam2_checkpoint_path`: Path to SAM2 model weights. The default is `"../checkpoints/sam2.1_hiera_large.pt"`.
  - `sam2_config_path`: Path to SAM2 configuration file. The default is `"configs/sam2.1/sam2.1_hiera_l.yaml"`.
  - `detect_model_weights`: Path to YOLOv8 detection model weights. In the billiard scene example, the path is `"det_weights/train_referee12_960.pt"`.
  - `load_inference_state_path`: Path to the preloaded memory bank. Default is `None`. If a preloaded memory bank is needed, provide the path to it (the preloaded memory bank is automatically generated by the `det_sam2_RT.py` script, as explained in its parameter section).
  - `temp_video_segments_pkl`: Default is `"./temp_output/video_segments.pkl"`, which is the path to the `video_segments` dictionary containing temporary segmentation inference results from the `VideoProcessor` class.
  - `temp_special_classes_detection_pkl`: Path to store detection results for special classes (i.e., categories where detection results do not need to be passed to SAM2 for segmentation inference, and once detected in any frame of the video, the `special_classes_detection` dictionary is considered complete). Default is `"./temp_output/special_classes_detection.pkl"`.
  - `visualize_result_dir`: Path for visualizing the post-processing results from the `VideoPostProcessor` class. If set to `None`, no visualization is performed. Default is `None`.

  

- **The `EvalDetSAM2PostProcess.eval_all_settings()` inference function requires the following parameters:**

  - `videos_dir`: Default is `"./data/Det-SAM2-Evaluation/videos"`, which is the folder containing the evaluation dataset videos.
  - `eval_jsonl_path`: Default is `"./data/Det-SAM2-Evaluation/only_test.jsonl"`, which is the path to the evaluation dataset annotation file in JSONL format.
  - `eval_output_dir`: Default is `"./eval_output/eval_result"`, which is the folder to store the evaluation results.

  In addition to these, other parameters should be provided as lists. For example, if you want to evaluate the script with detection confidence values of `0.6`, `0.8`, and `0.9`, you should pass `[0.6, 0.8, 0.9]` for `detect_confidence_list`.











