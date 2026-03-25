import sys, os, ctypes, numbers
print("[PYDBG] exec:", sys.executable)
print("[PYDBG] ver :", sys.version)
print("[PYDBG] path0:", sys.path[:5])
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# for lib in ("libnvinfer.so.8", "libnvinfer_plugin.so.8", "libcudart.so.11.8.89", "libcudnn.so.8"):
#     try:
#         ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
#         print(f"[DL] preloaded {lib}")
#     except OSError as e:
#         print(f"[DL] WARN: {lib} preload failed: {e}")

import tensorrt as trt

from cuda import cudart
cudart.cudaFree(0)

import torch
import copy
import numpy as np
from abc import ABC, abstractmethod

import rospy
import message_filters
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from tf.transformations import quaternion_from_euler
from autohyu_msgs.msg import DetectObjects3D, DetectObject3D, ObjectDimension, Object3DState

import rospkg
import cv2
import sys

import threading
from collections import deque

sys.path.append(".")

import time


def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes

class BaseBBoxCoder(ABC):
    """Base bounding box coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""



class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
       
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list

class BEVFormerTool(NMSFreeCoder):
    def __init__(
        self,
        pc_range,
        post_center_range=None,
        max_num=100,
        score_threshold=None,
        num_classes=10,
        voxel_size=None
    ):
        super().__init__(
            pc_range=pc_range,
            voxel_size=voxel_size,
            post_center_range=post_center_range,
            max_num=max_num,
            score_threshold=score_threshold,
            num_classes=num_classes,
        )

    @torch.no_grad()
    def get_bboxes(self, preds_dicts, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]

            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]["box_type_3d"](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]

            ret_list.append([bboxes, scores, labels])

        return ret_list
    
    @torch.no_grad()
    def bbox3d2result(self, bboxes, scores, labels, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu()
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    @torch.no_grad()
    def post_process(self, outputs_classes, outputs_coords):
            dic = {"all_cls_scores": outputs_classes, "all_bbox_preds": outputs_coords}
            result_list = self.get_bboxes(dic, rescale=True)

            return [
                {
                    "pts_bbox": self.bbox3d2result(bboxes, scores, labels)
                    for bboxes, scores, labels in result_list
                }
            ]

def get_logger(level=trt.Logger.INTERNAL_ERROR):
    TRT_LOGGER = trt.Logger(level)
    return TRT_LOGGER

def create_engine_context(trt_model, trt_logger):
    print("[TRT] py tensorrt:", trt.__version__)
    print("[TRT] trying to load engine:", trt_model)
    assert os.path.exists(trt_model)

    with open(trt_model, "rb") as f, trt.Runtime(trt_logger) as runtime:
        blob = f.read()
        print("[TRT] engine bytes:", len(blob))
        engine = runtime.deserialize_cuda_engine(blob)

    if engine is None:
        raise RuntimeError("deserialize_cuda_engine returned None (plugin/version mismatch?)")

    print("[TRT] engine bindings:", engine.num_bindings)
    for i, name in enumerate(engine):
        print("  -", i, name, "is_input=", engine.binding_is_input(name),
              "dtype=", engine.get_binding_dtype(name), "shape=", engine.get_binding_shape(name))

    if engine.num_optimization_profiles > 0:
        print("[TRT] num opt profiles:", engine.num_optimization_profiles)

    context = engine.create_execution_context_without_device_memory()
    if context is None:
        raise RuntimeError("create_execution_context_without_device_memory returned None")

    if engine.num_optimization_profiles > 0:
        ok = context.set_optimization_profile_async(0, 0)  # cudaStream 0
        print("[TRT] set_optimization_profile_async(0) ->", ok)

    dmem_size = engine.device_memory_size
    print("[TRT] engine.device_memory_size:", dmem_size)
    err, dmem_ptr = cudart.cudaMalloc(dmem_size)
    assert err == cudart.cudaError_t.cudaSuccess, f"cudaMalloc({dmem_size}) failed: {err}"
    context.device_memory = int(dmem_ptr)

    return engine, context, dmem_ptr

class HostDeviceMem(object):
    def __init__(self, name, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.name = name
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return (
            "Name:\n"
            + str(self.name)
            + "\nHost:\n"
            + str(self.host)
            + "\nDevice:\n"
            + str(self.device)
            + "\n"
        )

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine, context, input_shapes):
    inputs, outputs, bindings = [], [], []

    # 1) 먼저 모든 input binding shape를 context에 설정
    for binding_id, name in enumerate(engine):
        if engine.binding_is_input(name):
            dims = tuple(input_shapes[name])
            ok = context.set_binding_shape(binding_id, dims)
            if not ok:
                raise RuntimeError(f"set_binding_shape failed: {name} -> {dims}")

    if not context.all_binding_shapes_specified:
        raise RuntimeError("Not all binding shapes specified (dynamic shape engine).")

    # 2) 이제 실제 binding shape를 context에서 읽어서 그 크기로 버퍼 할당
    for binding_id, name in enumerate(engine):
        dims = tuple(context.get_binding_shape(binding_id))  # 실제 shape
        if any(d <= 0 for d in dims):
            raise RuntimeError(f"Invalid binding shape for {name}: {dims}")

        size = int(trt.volume(dims))
        dtype = trt.nptype(engine.get_binding_dtype(name))
        if dtype != np.float32:
            raise RuntimeError(f"{name} dtype={dtype}, expected FP32")

        host_mem = np.empty(size, dtype=dtype)
        err, device_ptr = cudart.cudaMalloc(host_mem.nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed for {name}, bytes={host_mem.nbytes}, err={err}")

        bindings.append(int(device_ptr))
        hdm = HostDeviceMem(name, host_mem, int(device_ptr))
        if engine.binding_is_input(name):
            inputs.append(hdm)
        else:
            outputs.append(hdm)

        print(f"[BUF] {name}: shape={dims}, bytes={host_mem.nbytes}")

    return inputs, outputs, bindings

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # HtoD
    for inp in inputs:
        err = cudart.cudaMemcpyAsync(
            inp.device,                                # dst (device ptr)
            int(inp.host.ctypes.data),                 # src (host ptr)
            inp.host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream
        )[0]
        assert err == cudart.cudaError_t.cudaSuccess

    cudart.cudaStreamSynchronize(stream)
    t1 = time.time()

    ok = context.execute_async_v2(bindings=bindings, stream_handle=stream)
    if not ok:
        raise RuntimeError("TensorRT execute_async_v2 failed")

    cudart.cudaStreamSynchronize(stream)
    t2 = time.time()

    # DtoH
    for out in outputs:
        err = cudart.cudaMemcpyAsync(
            int(out.host.ctypes.data),                 # dst (host ptr)
            out.device,                                # src (device ptr)
            out.host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream
        )[0]
        assert err == cudart.cudaError_t.cudaSuccess

    cudart.cudaStreamSynchronize(stream)

    return outputs, t2 - t1


def impad(img,
          *,
          shape=None,
          padding=None,
          pad_val=0,
          padding_mode='constant'):
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (h, w). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
                with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the
                last value on the edge. For example, padding [1, 2, 3, 4]
                with 2 elements on both sides in reflect mode will result
                in [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last
                value on the edge. For example, padding [1, 2, 3, 4] with
                2 elements on both sides in symmetric mode will result in
                [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        border_type[padding_mode],
        value=pad_val)

    return img


class BEVFormerNode:
    def __init__(self):
        rp = rospkg.RosPack()
        PKG = 'bevformer_pkg'
        pkg_path = rp.get_path(PKG)

        so_path   = os.path.join(pkg_path, 'lib', 'libmmdeploy_tensorrt_ops.so')
        trt_path  = os.path.join(pkg_path, 'models', 'bevformer_latest.trt')
        calib_path = os.path.join(pkg_path, 'data', 'lidar2img.npy')
        rospy.loginfo(so_path)

        rospy.init_node('3dod_node', anonymous=True)
        self.camera_topics = rospy.get_param('~camera_topics',
                                             [f'/camera_{i}_undistorted/compressed' for i in range(1, 7)])
        self.odom_topic = rospy.get_param('~odom_topic', '/novatel/oem7/odom')

        self.pub_boxes  = rospy.Publisher("bevformer/boxes", MarkerArray, queue_size=1)
        self.pub_custom = rospy.Publisher("bevformer/detect_objects", DetectObjects3D, queue_size=1)

        # --- overlay publishers (6 cameras) ---
        self.pub_overlay = []
        for i in range(1, 7):
            topic = f"bevformer/overlay/camera_{i}/compressed"
            self.pub_overlay.append(rospy.Publisher(topic, CompressedImage, queue_size=1))

        self.vis_frame = rospy.get_param("~vis_frame", "base_link")
        self.score_thr = rospy.get_param("~score_thr", 0.8)

        # --- 실시간 튜닝 파라미터 ---
        self.enable_overlay   = rospy.get_param("~enable_overlay", True)
        self.overlay_every_n  = int(rospy.get_param("~overlay_every_n", 1))  # 1=매프레임, 2=2프레임에 1번
        self._frame_idx = 0

        # --- Odom 최신값 캐시 (sync에서 제외) ---
        self._odom_lock = threading.Lock()
        self._latest_odom = None
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self._odom_cb, queue_size=50)

        # --- TensorRT / plugin load ---
        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        print(f"[DL] loaded plugin: {so_path}")

        self.tool = BEVFormerTool(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            score_threshold=self.score_thr,
            voxel_size=[0.2, 0.2, 8],
            num_classes=4
        )

        TRT_LOGGER = get_logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        self.engine, self.context, self._ctx_dmem = create_engine_context(trt_path, TRT_LOGGER)

        err, self.stream = cudart.cudaStreamCreate()
        assert err == cudart.cudaError_t.cudaSuccess

        self.lidar2img = np.load(calib_path)

        self.img_norm_cfg = dict(
            mean=np.array([123.675, 116.28, 103.53]),
            std=np.array([58.395, 57.12, 57.375])
        )

        self.input_shapes = dict(
            image=["batch_size", "cameras", 3, "img_h", "img_w"],
            prev_bev=["bev_h*bev_w", "batch_size", "dim"],
            use_prev_bev=[1],
            can_bus=[18],
            lidar2img=["batch_size", "cameras", 4, 4],
        )

        self.output_shapes = dict(
            bev_embed=["bev_h*bev_w", "batch_size", "dim"],
            outputs_classes=["cameras", "batch_size", "num_query", "num_classes"],
            outputs_coords=["cameras", "batch_size", "num_query", "code_size"],
        )

        self.bev_h = 50
        self.bev_w = 50
        self.dim = 256

        self.default_shapes = dict(
            batch_size=1,
            img_h=544,
            img_w=960,
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            dim=self.dim,
            num_query=900,
            num_classes=4,
            code_size=10,
            cameras=6,
        )

        self.output_shapes_eval = self._eval_shapes(self.output_shapes, self.default_shapes)
        self.input_shapes_eval  = self._eval_shapes(self.input_shapes, self.default_shapes)

        self.inputs, self.outputs, self.bindings = allocate_buffers(
            self.engine, self.context, input_shapes=self.input_shapes_eval
        )

        self.prev_frame_info = {"prev_pos": 0, "prev_angle": 0}

        D = self.default_shapes
        shape = (D["bev_h"] * D["bev_w"], 1, D["dim"])
        self.prev_bev = np.zeros(shape, dtype=np.float32)
        self.first = True

        # --- 카메라 6개만 sync (odom은 제외) ---
        #   CompressedImage는 buff_size 부족 시 drop/지연이 생길 수 있어 키우는 것을 권장합니다.
        cam_subs = [
            message_filters.Subscriber(topic, CompressedImage, queue_size=1, buff_size=2**24)
            for topic in self.camera_topics
        ]
        self.ts = message_filters.ApproximateTimeSynchronizer(cam_subs, queue_size=30, slop=0.1)
        self.ts.registerCallback(self._cam_sync_cb)

        # --- 최신 1개 큐 (drop old) + worker thread ---
        self._q = deque(maxlen=1)
        self._q_lock = threading.Lock()
        self._q_cv = threading.Condition(self._q_lock)
        self._running = True

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo("BEVFormerNode initialized (async worker mode).")

    def _eval_shapes(self, shapes_dict, defaults):
        out = {}
        local_env = dict(defaults)
        for name, shape in shapes_dict.items():
            arr = []
            for s in shape:
                if isinstance(s, str):
                    arr.append(eval(s, {}, local_env))
                else:
                    arr.append(s)
            out[name] = arr
        return out
    
    def _bbox3d_to_corners(self, box7):
        """
        box7: [cx, cy, cz_bottom, w, l, h, yaw]  (yaw: z축 회전)
        return: (8,3) corners in same frame as box (x,y,z)
        """
        cx, cy, czb, w, l, h, yaw = [float(x) for x in box7]
        cz = czb + 0.5 * h

        # local corners (center 기준)
        # x: length 방향, y: width 방향
        x_c = l * 0.5
        y_c = w * 0.5
        z_c = h * 0.5

        # 8 corners in box local frame
        corners = np.array([
            [ x_c,  y_c,  z_c],
            [ x_c, -y_c,  z_c],
            [-x_c, -y_c,  z_c],
            [-x_c,  y_c,  z_c],
            [ x_c,  y_c, -z_c],
            [ x_c, -y_c, -z_c],
            [-x_c, -y_c, -z_c],
            [-x_c,  y_c, -z_c],
        ], dtype=np.float32)

        # yaw rotation around z
        c = np.cos(yaw)
        s = np.sin(yaw)
        R = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        corners = corners @ R.T
        corners[:, 0] += cx
        corners[:, 1] += cy
        corners[:, 2] += cz
        return corners


    def _project_lidar_to_img(self, pts3, lidar2img_4x4, depth_min=0.5):
        """
        pts3: (N,3)
        lidar2img_4x4: (4,4)
        return: uv (N,2), valid (N,), depth (N,)
        """
        N = pts3.shape[0]
        pts4 = np.concatenate([pts3, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N,4)
        proj = (lidar2img_4x4 @ pts4.T).T  # (N,4)

        depth = proj[:, 2].astype(np.float32)
        valid = depth > float(depth_min)

        uv = np.zeros((N, 2), dtype=np.float32)
        uv[valid, 0] = proj[valid, 0] / depth[valid]
        uv[valid, 1] = proj[valid, 1] / depth[valid]
        return uv, valid, depth


    def _draw_projected_bbox(self, img_bgr, uv, valid, color, thickness=2):
        """
        uv: (8,2) float
        valid: (8,) bool  (depth_min 통과 여부)
        """
        H, W = img_bgr.shape[:2]

        # bbox edge indices for 8 corners (top:0-3, bottom:4-7)
        edges = [
            (0,1), (1,2), (2,3), (3,0),   # top
            (4,5), (5,6), (6,7), (7,4),   # bottom
            (0,4), (1,5), (2,6), (3,7)    # verticals
        ]

        rect = (0, 0, W, H)  # clipLine expects (x, y, w, h)

        for i, j in edges:
            # 둘 다 depth 유효(카메라 앞)일 때만 선분을 그림
            if not (valid[i] and valid[j]):
                continue

            x1, y1 = float(uv[i, 0]), float(uv[i, 1])
            x2, y2 = float(uv[j, 0]), float(uv[j, 1])

            # 너무 큰 outlier는 안전하게 스킵 (분모 폭발 잔재 방지)
            # 필요 시 계수(5)를 더 키워도 됩니다.
            if (abs(x1) > 5*W) or (abs(x2) > 5*W) or (abs(y1) > 5*H) or (abs(y2) > 5*H):
                continue

            p1 = (int(round(x1)), int(round(y1)))
            p2 = (int(round(x2)), int(round(y2)))

            # 선분이 화면과 교차하면 교차 구간으로 잘라서 반환해줌
            ok, q1, q2 = cv2.clipLine(rect, p1, p2)
            if not ok:
                continue

            cv2.line(img_bgr, q1, q2, color, thickness, lineType=cv2.LINE_AA)

    def _encode_compressed(self, img_bgr, stamp, frame_id, fmt="jpeg", quality=80):
        msg = CompressedImage()
        msg.header = Header(stamp=stamp, frame_id=frame_id)
        msg.format = fmt

        if fmt.lower() in ("jpg", "jpeg"):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ok, buf = cv2.imencode(".jpg", img_bgr, encode_param)
        else:
            ok, buf = cv2.imencode(".png", img_bgr)

        if not ok:
            raise RuntimeError("cv2.imencode failed")
        msg.data = buf.tobytes()
        return msg
    
    def _decode_compressed(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img_bgr
    
    def _bbox_results_to_detectobjects(self, bbox_list):
        msg = DetectObjects3D()
        msg.header = Header(frame_id=self.vis_frame, stamp=rospy.Time.now())

        if not bbox_list:
            return msg

        res    = bbox_list[0]["pts_bbox"]
        boxes  = res["boxes_3d"].cpu().numpy()   # (N, 7 or 9): [cx,cy,cz_bottom,w,l,h,yaw,(vx,vy)]
        scores = res["scores_3d"].cpu().numpy()  # (N,)
        labels = res["labels_3d"].cpu().numpy()  # (N,)

        obj_id = 0
        for i in range(boxes.shape[0]):
            if scores[i] < self.score_thr:
                continue

            cx, cy, cz_bottom, w, l, h, yaw = boxes[i, :7]

            yaw = -float(yaw) - np.pi/2 # yaw 값 보정
            yaw = float(yaw)

            z_center = float(cz_bottom + 0.5 * h)
            vx, vy = 0.0, 0.0
            if boxes.shape[1] >= 9:
                vx = float(boxes[i, 7])
                vy = float(boxes[i, 8])

            dobj = DetectObject3D()
            dobj.id                = int(obj_id); obj_id += 1
            dobj.confidence_score  = float(scores[i])
            dobj.classification    = int(labels[i])

            dim = ObjectDimension()
            dim.length = float(l)
            dim.width  = float(w)
            dim.height = float(h)

            st = Object3DState()
            st.header       = msg.header
            st.x, st.y, st.z = float(cx), float(cy), z_center
            st.vx, st.vy, st.vz = vx, vy, 0.0
            st.ax = st.ay = st.az = 0.0
            st.roll = st.pitch = 0.0
            st.yaw = yaw
            st.roll_rate = st.pitch_rate = st.yaw_rate = 0.0

            dobj.dimension = dim
            dobj.state     = st

            msg.object.append(dobj)

        return msg
        
    def _bbox_results_to_markers(self, bbox_list):
        ma = MarkerArray()
        if not bbox_list:
            return ma

        res = bbox_list[0]["pts_bbox"]  # {"boxes_3d", "scores_3d", "labels_3d"}
        boxes = res["boxes_3d"].cpu().numpy()     # (N, 7 or 9)
        scores = res["scores_3d"].cpu().numpy()   # (N,)
        labels = res["labels_3d"].cpu().numpy()   # (N,)

        # 간단 팔레트
        palette = [
            (0.1, 0.8, 0.2, 0.9),
            (0.1, 0.6, 1.0, 0.9),
            (1.0, 0.7, 0.1, 0.9),
            (1.0, 0.2, 0.3, 0.9),
        ]

        now = rospy.Time.now()
        mid = 0
        for i in range(boxes.shape[0]):
            if scores[i] < self.score_thr:
                continue

            cx, cy, cz_bottom, w, l, h, yaw = boxes[i, :7]
            z_center = float(cz_bottom + 0.5 * h)  # 바닥→중심 보정
            qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))

            m = Marker()
            m.header = Header(frame_id=self.vis_frame, stamp=now)
            m.ns = "bevformer_boxes"
            m.id = mid; mid += 1
            m.type = Marker.CUBE
            m.action = Marker.ADD

            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = z_center
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw

            # 길이/너비/높이 매핑 (x=length, y=width, z=height)
            m.scale.x = float(l)
            m.scale.y = float(w)
            m.scale.z = float(h)

            r, g, b, a = palette[int(labels[i]) % len(palette)]
            m.color = ColorRGBA(r=r, g=g, b=b, a=a)
            m.lifetime = rospy.Duration(0.5)  # 주기 갱신

            ma.markers.append(m)

        return ma

    # ----------------------------
    # ✅ lightweight callbacks
    # ----------------------------
    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self._latest_odom = msg

    def _cam_sync_cb(self, *cam_msgs):
        # 1) 최신 odom 스냅샷 확보
        with self._odom_lock:
            odom = self._latest_odom

        if odom is None:
            return

        # 2) 최신 1개만 유지되는 큐에 push하고 바로 return (heavy work 금지)
        stamp = cam_msgs[0].header.stamp  # 대표 stamp (원하시면 다른 기준으로 바꿔도 됩니다)
        with self._q_cv:
            self._q.append((stamp, cam_msgs, odom))
            self._q_cv.notify()

    # ----------------------------
    # ✅ worker loop
    # ----------------------------
    def _worker_loop(self):
        while self._running and (not rospy.is_shutdown()):
            with self._q_cv:
                while self._running and len(self._q) == 0 and (not rospy.is_shutdown()):
                    self._q_cv.wait(timeout=0.2)

                if (not self._running) or rospy.is_shutdown():
                    break

                # "최신"만 꺼냄 (deque maxlen=1이므로 이미 old는 drop됨)
                stamp, cam_msgs, odom = self._q.pop()

            t0 = time.time()
            try:
                self._process_one(stamp, cam_msgs, odom)
            except Exception as e:
                import traceback as tb
                rospy.logerr("Worker exception: %s\n%s", e, tb.format_exc())

            dt_ms = (time.time() - t0) * 1000.0
            rospy.loginfo_throttle(1.0, f"[worker] process ms: {dt_ms:.1f}")

    # ----------------------------
    # ✅ heavy work (기존 callback() 내용 이관)
    # ----------------------------
    @torch.no_grad()
    def _process_one(self, stamp, cam_msgs, odom):
        # ---- (0) can_bus 계산 (기존 callback 그대로) ----
        pos = [
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            odom.pose.pose.position.z
        ]
        q = [
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ]

        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[:3] = copy.deepcopy(pos)
        can_bus[3:7] = copy.deepcopy(q)

        qx, qy, qz, qw = q
        yaw = np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )
        patch_angle = np.degrees(yaw)
        if patch_angle < 0:
            patch_angle += 360

        can_bus[-2] = np.radians(patch_angle)
        can_bus[-1] = patch_angle

        D = self.default_shapes
        B, C, H, W = D["batch_size"], D["cameras"], D["img_h"], D["img_w"]

        # ROS → BEVFormer 입력 순서 재정렬
        camera_reorder_indices = [2, 3, 5, 1, 0, 4]  # 0-indexed
        cams = list(cam_msgs)
        reordered_cams = [cams[i] for i in camera_reorder_indices]

        # ---- (1) 이미지 전처리 (기존 callback 그대로) ----
        imgs = []
        vis_imgs = []
        mean = self.img_norm_cfg['mean']
        std  = self.img_norm_cfg['std']
        divisor = 32

        for m in reordered_cams:
            img_bgr0 = self._decode_compressed(m)  # 기존 helper 사용

            # overlay용 동일 resize+pad
            y_size = int(img_bgr0.shape[0] * 0.5)
            x_size = int(img_bgr0.shape[1] * 0.5)
            size = (x_size, y_size)
            img_vis = cv2.resize(img_bgr0, size, interpolation=cv2.INTER_LINEAR)

            pad_h = int(np.ceil(img_vis.shape[0] / divisor)) * divisor
            pad_w = int(np.ceil(img_vis.shape[1] / divisor)) * divisor
            img_vis = impad(img_vis, shape=(pad_h, pad_w), pad_val=0)
            vis_imgs.append(img_vis)

            # inference용 normalize
            img = img_bgr0.copy().astype(np.float32)
            mean_ = np.float64(mean.reshape(1, -1))
            stdinv = 1 / np.float64(std.reshape(1, -1))

            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            cv2.subtract(img, mean_, img)
            cv2.multiply(img, stdinv, img)

            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
            img = impad(img, shape=(pad_h, pad_w), pad_val=0)
            img = img.transpose(2, 0, 1)  # CHW
            imgs.append(img)

        image = np.stack(imgs, axis=0)[None, ...].astype(np.float32)  # [1,6,3,H,W]

        use_prev_bev = np.array([1.0], dtype=np.float32)
        if self.first:
            use_prev_bev = np.array([0.0], dtype=np.float32)
            self.first = False

        if use_prev_bev[0] == 1:
            can_bus[:3] -= self.prev_frame_info["prev_pos"]
            can_bus[-1] -= self.prev_frame_info["prev_angle"]
        else:
            can_bus[:3] -= 0
            can_bus[-1] -= 0

        # ---- (2) TRT 입력 host 버퍼 채우기 ----
        for inp in self.inputs:
            if inp.name == "image":
                inp.host[:] = image.reshape(-1)
            elif inp.name == "prev_bev":
                inp.host[:] = self.prev_bev.reshape(-1).astype(np.float32)
            elif inp.name == "use_prev_bev":
                inp.host[:] = use_prev_bev.reshape(-1)
            elif inp.name == "can_bus":
                inp.host[:] = can_bus.reshape(-1)
            elif inp.name == "lidar2img":
                inp.host[:] = self.lidar2img.reshape(-1).astype(np.float32)
            else:
                raise RuntimeError(f"Cannot find input name {inp.name}.")

        # ---- (3) TRT inference ----
        trt_out_list, t = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )

        trt_map = {
            o.name: o.host.reshape(*self.output_shapes_eval[o.name]).astype(np.float32)
            for o in trt_out_list
        }

        self.prev_bev = trt_map.pop("bev_embed")
        self.prev_frame_info["prev_pos"] = np.array(pos, dtype=np.float32)
        self.prev_frame_info["prev_angle"] = patch_angle

        trt_outputs = {k: torch.from_numpy(v) for k, v in trt_map.items()}
        bbox_list = self.tool.post_process(**trt_outputs)

        if not bbox_list:
            return

        # yaw 보정(기존 코드 유지)
        res = bbox_list[0]["pts_bbox"]
        boxes = res["boxes_3d"]
        if boxes.numel() > 0:
            boxes[:, 6] = -boxes[:, 6] - np.pi/2

        # ---- (4) overlay (옵션/저주기) ----
        self._frame_idx += 1
        do_overlay = self.enable_overlay and (self._frame_idx % self.overlay_every_n == 0)

        if do_overlay:
            boxes_np  = res["boxes_3d"].cpu().numpy()
            scores_np = res["scores_3d"].cpu().numpy()
            labels_np = res["labels_3d"].cpu().numpy()

            palette_bgr = [
                ( 50, 200,  50),
                (255, 160,  50),
                ( 50, 150, 255),
                ( 60,  60, 255),
            ]

            # stamp는 "입력 프레임 stamp" 유지 권장 (Time.now()로 바꾸면 동기화가 흔들립니다)
            out_stamp = stamp

            lidar2img = self.lidar2img
            if lidar2img.ndim == 4:
                lidar2img_cams = lidar2img[0]   # (6,4,4)
            elif lidar2img.ndim == 3:
                lidar2img_cams = lidar2img      # (6,4,4)
            else:
                raise RuntimeError(f"Unexpected lidar2img shape: {lidar2img.shape}")

            depth_min = 0

            for net_i in range(6):
                img_vis = vis_imgs[net_i].copy()
                P = lidar2img_cams[net_i].astype(np.float32)
                HH, WW = img_vis.shape[:2]

                for bi in range(boxes_np.shape[0]):
                    if scores_np[bi] < self.score_thr:
                        continue

                    box7 = boxes_np[bi, :7].copy()

                    # center gating
                    cx, cy, czb, w_box, l_box, h_box, yaw_ = [float(x) for x in box7]
                    cz = czb + 0.5 * h_box
                    center = np.array([[cx, cy, cz]], dtype=np.float32)

                    uv_c, valid_c, depth_c = self._project_lidar_to_img(center, P, depth_min=depth_min)
                    if not valid_c[0]:
                        continue
                    u0, v0 = float(uv_c[0,0]), float(uv_c[0,1])
                    if not (-50 <= u0 < WW + 50 and -50 <= v0 < HH + 50):
                        continue

                    corners = self._bbox3d_to_corners(box7)
                    uv, valid, depth = self._project_lidar_to_img(corners, P, depth_min=depth_min)
                    if int(valid.sum()) < 6:
                        continue

                    color = palette_bgr[int(labels_np[bi]) % len(palette_bgr)]
                    self._draw_projected_bbox(img_vis, uv, valid, color=color, thickness=2)

                cam_orig_i = camera_reorder_indices[net_i]  # net->orig
                out_msg = self._encode_compressed(
                    img_vis,
                    stamp=out_stamp,
                    frame_id=f"camera_{cam_orig_i+1}",
                    fmt="jpeg",
                    quality=80
                )
                self.pub_overlay[cam_orig_i].publish(out_msg)

        # ---- (5) markers + custom msg publish ----
        ma = self._bbox_results_to_markers(bbox_list)
        self.pub_boxes.publish(ma)

        det_msg = self._bbox_results_to_detectobjects(bbox_list)
        self.pub_custom.publish(det_msg)

        rospy.loginfo_throttle(1.0, f"Published {len(ma.markers)} boxes")

    # ----------------------------
    # ✅ shutdown
    # ----------------------------
    def _on_shutdown(self):
        self._running = False
        try:
            with self._q_cv:
                self._q_cv.notify_all()
        except Exception:
            pass

        try:
            if hasattr(self, "worker") and self.worker.is_alive():
                self.worker.join(timeout=1.0)
        except Exception:
            pass

    def spin(self):
        rospy.loginfo("BEVFormerNode spinning...")
        rospy.spin()


if __name__ == "__main__":
    node = BEVFormerNode()
    node.spin()