### Det-SAM2-pipeline

Det-SAM2项目是一个基于Segment Anything Model 2 分割模型（[SAM2](https://github.com/facebookresearch/sam2)），利用YOLOv8检测模型自动为SAM2添加提示，再使用后处理部分对SAM2分割结果进行专有场景业务判断（本项目实现于台球场景）的，无需人工干预自动视频物体追踪pipeline。

对于SAM2适配预测器，我们的核心贡献有：

- 实现了无需人工干预提示交互的自提示视频实例分割pipeline（Det-SAM2-pipeline）。其支持对特定类别（由自定义检测模型决定）以**视频流**进行推理分割，并且以SAM2原有的精度返回分割结果以支持后处理业务层的使用。
- 我们额外实现了在**推理追踪过程中添加新的类别**而不中断推理状态的功能。
- 我们的pipeline允许将一段视频上推理过后的记忆库应用在新的视频上，我们称其为预加载记忆库（preload memory bank）。它可以利用上一段视频中推理分析的记忆（物体类别/形态/运动状态）辅助其在新的视频中执行类似的推理而**不需要在新的视频中添加任何条件提示**。
- 我们实现了在Det-SAM2-pipeline上**恒定的显存与内存开销**，从而支持一次性推理无限长的视频。

如果您在着手SAM2的工程优化，我们非常欢迎您参考我们Det-SAM2的实现方式。同时我们使用Det-SAM2构建了一个完整pipeline，其在业务场景（台球场景）实现了进球判断、球间碰撞与边界反弹检测。此前非SAM2的传统追踪算法几乎无法实现高速移动的台球场景下这三个条件的准确判断。



注：我们开源的脚本为**全中文注释**以便开发，不影响功能使用，请您谅解。如果参考脚本功能实现时有需要，请使用ChatGPT等将注释翻译成您的语言。



### Installation

我们项目基于SAM2.1版本构建，我们的环境依赖几乎一样，您可以直接运行SAM2.1的环境安装方式来部署它：https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation

除此之外可能有一些额外的包需要独立独立安装，请根据报错提示安装他们（都是一些常见的包，不会很多）

或者您可以直接进入AutoDL中我们已经发布的镜像https://www.codewithgpu.com/i/motern88/Det-SAM2/Det-SAM2，这样您将可以直接上手几乎无需任何环境配置。

我们项目的大部分可执行文件位于segment-anything-2根目录下的det_sam2_inference文件夹：

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

同时Det-SAM2相比SAM2的其他改动涉及（我们的改动只添加新的功能，不会失去任何SAM2.1中官方已有功能实现）：

```python
segment-anything-2/sam2:
├──modeling
│   ├──sam2_base.py
├──utils
│   ├──misc.py
sam2_video_predictor.py
```

#### Checkpoints

我们使用SAM2.1权重:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

检测模型可以使用您任意的YOLOv8权重，或者先使用我们在台球场景下训练的台球检测模型:

- [Det-SAM2-YOLO8-Weight](https://huggingface.co/ATA-space/Det-SAM2/tree/main/det_weights)



### Getting Started

以下脚本均是默认以台球场景示例进行后处理判断，如果您只需要使用Det-SAM2框架，只使用det_sam2_RT即可。

1.单独执行Det-SAM2分割mask预测脚本和后处理判断脚本

使用检测模型自动提供提示，SAM2接收提示对视频进行分割预测：

```cmd
python det_sam2_inference/det_sam2_RT.py
```

`det_sam2_RT.py`定义`VideoProcessor`类（**参数设置见下文**），主要使用其中的`VideoProcessor.run()`方法

在执行过`det_sam2_RT.py`，并保存`self.video_segments`分割结果的字典后，使用后处理对已分割mask进行业务判断：

```cmd
python det_sam2_inference/postprocess_det_sam2.py
```

`postprocess_det_sam2.py`定义`VideoPostProcessor`类（**参数设置见下文**），主要使用其中的`VideoPostProcessor.run()`方法。



2.直接执行全流程脚本（可以支持恒定显存与内存开销），异步并行地推理分割mask与后处理判断

执行全流程脚本一次性推理长视频：

```cmd
python det_sam2_inference/Det_SAM2_pipeline.py
```

`Det_SAM2_pipeline.py`定义`DetSAM2Pipeline`类（**参数设置见下文**），该类将分割主干`VideoPostProcessor`和后处理`VideoPostProcessor`的关键功能以异步并行的方式处理实时视频流推理，主要使用其中的`DetSAM2Pipeline.inference()`方法。



3.执行自动评估脚本，用于尝试各种参数组合：

```cmd
python det_sam2_inference/eval_det-sam2.py
```

`eval_det-sam2.py`定义`EvalDetSAM2PostProcess`类（**参数设置见下文**），该类对多种候选参数组合循环推理整个评估数据集，在每个样本推理中，分割主干`VideoPostProcessor.run()`和后处理`VideoPostProcessor.run()`串行推理。最终收集评估结果写入到`eval_results.json`，该脚本主要执行`EvalDetSAM2PostProcess.eval_all_settings()`方法。

在完成eval_results.json写入后，可以进行各个参数评估结果的可视化展示：

```cmd
python det_sam2_inference/eval_output/eval_result/result_visualize.py
```



#### Parameter Parsing

下面对三个重要脚本（`det_sam2_RT.py`、`postprocess_det_sam2.py`、`Det_SAM2_pipeline.py`和`eval_det-sam2.py`）的功能实现类及对应类中主函数进行参数介绍。

- 位于`det_sam2_RT.py`中的`VideoProcessor`类初始化时需要传入参数：

  `output_dir` : 渲染后结果的保存路径，默认`"./temp_output/det_sam2_RT_output"`。

  `sam2_checkpoint` : SAM2.1模型权重，例如`"../checkpoints/sam2.1_hiera_large.pt"`。

  `detect_model_weights` : 你自定义训练好特定类别的yolov8检测模型权重，在我们的台球场景示例中为`"../checkpoints/sam2.1_hiera_large.pt"`。

  `detect_confidence` : yolo检测模型置信度，在我们的示例中为`0.85`。

  `skip_classes` : 如果你的YOLO模型检测类别中有你不需要的，你不希望这些类别被作为prompt传递给SAM2。假设你希望检测模型跳过11、14、15、19类别（取决于你自己的检测模型类别ID），则设置`skip_classes={11, 14, 15, 19}`。

  `vis_frame_stride` : 间隔几帧渲染一次SAM2分割结果，设置为`-1`时不进行渲染，默认为`-1`。

  `visualize_prompt` : 是否可视化交互帧/条件帧（即存在检测模型提供条件提示的帧），用于检查检测模型是否提供了正确的条件提示，默认为`False`。

  `frame_buffer_size` : 我们实现的视频流推理不会每接收一帧就推理一帧，而是考虑推理效率选择累积一定数量的视频帧再统一推理。该参数控制的是视频帧累积缓存的容量，我们暂时设置每累积30帧进行一次SAM2推理，即默认`frame_buffer_size=30`。

  `detect_interval` : 我们的检测模型不会为每一帧都添加提示，而是间隔一定帧数才执行一次检测（为SAM2添加一次条件帧）。默认值是`30`——视频流中每30帧进行一次检测。当`detect_interval=-1`时不进行检测，但是推理中必须有至少一帧为条件帧。当推理加载在预加载记忆库时（preload memory bank）由于记忆库中均为条件帧，检测间隔可以为`-1`，后续不再添加任何条件帧，完全依靠预加载的记忆库的条件帧执行推理。总而言之，该参数的作用是控制条件帧出现的间隔。

  `max_frame_num_to_track` : SAM2视频推理会执行`SAM2VideoPredictor.propagate_in_video`函数，将根据所有条件帧，在所有非条件帧中传播推理以预测非条件帧中的分割mask。然而我们选择有限的传播推理长度，我们认为过去很久的视频帧已经有足够多的信息不需要被再次推理了。该参数控制视频传播推理的最大长度，默认`max_frame_num_to_track=60`，`max_frame_num_to_track`必须大于等于两倍的`frame_buffer_size`才能保证每一帧都至少可能在下一次传播推理过程中被修正。

  `max_inference_state_frames` : 该参数是记忆库资源占用优化的一部分，它实现了仅保留有限的记忆库帧数，超过这个最大记忆库保留帧数的旧的帧将会被释放。`max_inference_state_frames` 不得小于`max_frame_num_to_track`，默认为`60`，您可以尝试在不超出显存的情况下尽可能增大该参数，它可以保留更多用于计算的信息。

  `load_inference_state_path` : 默认为`None`，如果需要加载预加载记忆库，则传入你的预加载记忆库路径（.pkl）。

  `save_inference_state_path` :  默认为`None`。如果你需要制作预加载记忆库以支持将该视频的推理记忆迁移到新的视频中，则可以传入`"output_inference_state/inference_state_frames.pkl"`。同时制作用于预加载的记忆库的推理时，需要设置足够的`max_inference_state_frames`容量以提供足够多且有效的记忆信息。



- `VideoProcessor.run()`推理函数需要传入参数：

  `video_path` : 传入mp4视频路径（与`frame_dir`二选一），默认为`None`。

  `frame_dir` : 传入包含视频帧的文件夹（与`video_path`二选一），默认为`None`。

  `output_video_segments_pkl_path` : 输出分割mask字典的pkl路径，该字典收集了SAM2输出的分割mask可以用于后处理等其他操作，默认为`"./temp_output/video_segments.pkl"`。

  `output_special_classes_detection_pkl_path` : 输出特殊类别检测结果的pkl路径。如果需要直接输出检测结果的而不需要经过SAM2分割预测，则该路径为特殊类别检测结果直接输出的pkl路径，默认为`"./temp_output/special_classes_detection.pkl"`。



- 位于`postprocess_det_sam2.py`中的`VideoPostProcessor`类初始化参数，该类为基于Det-SAM2输出分割结果的台球场景的后处理类，用于判断球进洞、球间碰撞和球撞击桌边反弹。该后处理类初始化参数与视频输入分辨率强相关，以下默认参数均是以适配1920*1080分辨率为准，具体场景样本见评估集中的示例样本[Det-SAM2/data/Det-SAM2-Evaluation](https://huggingface.co/ATA-space/Det-SAM2/tree/main/data/Det-SAM2-Evaluation)：

  `pot_distance_threshold` : 默认`100`。用于判断是否处于袋口附近的阈值，增大可以提供更大的判定范围。

  `pot_velocity_threshold` : 默认`0.9`。用于判定进球速度向量的方向阈值，增大可以更宽松地允许球速度方向偏离洞口。

  `ball_distance_threshold` : 默认`120`。球间碰撞的距离判定阈值，两球在这个距离内才进行碰撞判定。

  `ball_velocity_threshold` : 默认`10`。球间碰撞的速度变化判断阈值，两球碰撞后速度变化情况(加速度)大于这个阈值时才可能认为发生碰撞。

  `table_margin` : 默认`100`。该值为桌边向中心延申一定长度创建的缓冲区域，该区域内可能发生桌边反弹。增大可以增加可能出现球在桌边反弹的判定范围。

  `rebound_velocity_threshold` : 默认`0.7`。判断球发生桌边反弹的速度阈值，增大可以更容易被判断成反弹（表示碰撞前后垂直速度分量大小的变化不超过这个阈值）。



- `VideoPostProcessor.run()`推理函数需要传入参数：

  `segments_dict_pkl` : 传入包含SAM2分割字典的PKL文件路径，由`det_sam2_RT.py`自动生成。

  `time_interval` : 默认`1.0`。计算速度向量的时间间隔，一般速度向量以每一帧前后间隔来统计。



- 位于`Det_SAM2_pipeline.py`中的`DetSAM2Pipeline`类初始化参数，该类在初始化中会实例化`VideoProcessor`和`VideoPostProcessor`类，并在`DetSAM2Pipeline.inference()`方法中异步并行地处理SAM2主干推理函数和后处理函数。

  `sam2_output_frame_dir` : SAM2分割可视化渲染后的结果保存文件夹，默认`"./temp_output/video_frames"`。

  `sam2_checkpoint_path` : SAM2的模型权重路径，默认`"../checkpoints/sam2.1_hiera_large.pt"`。

  `sam2_config_path` : SAM2的配置文件路径，默认`"configs/sam2.1/sam2.1_hiera_l.yaml"`。

  `detect_model_weights` : yolov8检测模型权重路径，在我们实现的台球场景示例中，该权重路径为`"det_weights/train_referee12_960.pt"`。

  `output_video_dir` : 后处理可视化输出文件夹，默认`"./pipeline_output"`。

  `load_inference_state_path` : 预加载内存库，默认为`None`，如果需要加载预加载记忆库，则传入你的预加载记忆库路径（预加载记忆库由`det_sam2_RT.py`脚本自动生成，详情见其参数解释）。

  `visualize_postprocessor` : 是否可视化后处理结果，默认为`False`。如果可视化后处理结果，则不能使用恒定的内存开销，`VideoProcessor`类需要保存`video_processor.inference_state["images"]`中所有的信息以支持后处理可视化。具体做法是`DetSAM2Pipeline.__init__()`中初始化`VideoProcessor`类时传入参数`max_inference_state_frames`最大保留帧数大于本次视频推理总帧数。

  注：在`DetSAM2Pipeline.__init__()`类初始化时会实例化`VideoProcessor`和`VideoPostProcessor`类，其中`VideoProcessor`在此处重新指定参数传入初始化设置，而`VideoPostProcessor`则以`postprocess_det_sam2.py`中的初始化设置为准。



- `DetSAM2Pipeline.inference()`推理函数需要传入参数：

  `video_source` : 视频源，可以传入本地MP4视频路径也可以传入rtsp url。

  `max_frames` : 接入视频流处理的最大帧数，处理达到这个帧数会手动结束推理，默认设置为`2000`。理论上可以处理无限帧，`DetSAM2Pipeline.inference()`已经实现了恒定的显存/内存占用。



- 位于`eval_det-sam2.py`中的`EvalDetSAM2PostProcess`类是用来在评估数据集中寻找`VideoProcessor`和`VideoPostProcessor`的最佳初始化参数组合的。其初始化参数为：

  `sam2_output_frame_dir` : 用来存储sam2的输出掩码的帧的临时文件夹，默认为`"./temp_output/det_sam2_RT_output"`。

  `sam2_checkpoint_path` : SAM2的模型权重，默认为`"../checkpoints/sam2.1_hiera_large.pt"`。

  `sam2_config_path` : SAM2的配置文件，默认为`"configs/sam2.1/sam2.1_hiera_l.yaml"`。

  `detect_model_weights` : yolov8检测模型权重，在我们实现的台球场景示例中，该权重路径为`"det_weights/train_referee12_960.pt"`。

  `load_inference_state_path` :  预加载内存库，默认为`None`，如果需要加载预加载记忆库，则传入你的预加载记忆库路径（预加载记忆库由`det_sam2_RT.py`脚本自动生成，详情见其参数解释）。

  `temp_video_segments_pkl` : 默认为`"./temp_output/video_segments.pkl"`，为`VideoProcessor`类临时输出分割推理结果的`video_segments`字典路径。

  `temp_special_classes_detection_pkl` :特殊类别（即检测模型中该类别的检测结果不需要送入SAM2进行分割推理，且一段视频中只要存在一帧该类别的检测结果`special_classes_detection`字典即收集完成）的检测结果输出结果的保存路径。默认为`"./temp_output/special_classes_detection.pkl"`。

  `visualize_result_dir` : `VideoPostProcessor`类后处理的可视化路径，传入`None`即不可视化，默认`None`。



- `EvalDetSAM2PostProcess.eval_all_settings()`推理函数需要传入参数：

  `videos_dir` : 默认为`"./data/Det-SAM2-Evaluation/videos"`，评估集视频文件夹。

  `eval_jsonl_path` : 默认为`"./data/Det-SAM2-Evaluation/only_test.jsonl"`，评估集标注文本jsonl文件路径。

  `eval_output_dir` : 默认为`"./eval_output/eval_result"`，评估集结果输出文件夹。

  除此之外其他参数，均以列表的形式传入。例如`detect_confidence_list`我想要评估脚本自动尝试`0.6`，`0.8`，`0.9`的参数组合，我需要传入`[0.6, 0.8, 0.9]`。
