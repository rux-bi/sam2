import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
# import torch_tensorrt
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import onnxruntime as ort
from torch.utils.benchmark import Timer, Measurement
from torchvision.transforms import Normalize, Resize, ToTensor

# matplotlib.use("TkAgg")  # Use TkAgg backend for interactive plotting
class SAM2FullModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = None
        self._bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        self.norm_mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1), requires_grad=False)
        self.norm_std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1), requires_grad=False)
        self.target_size = (1024, 1024)
        self.resize_op = Resize(self.target_size)
        
    def _resize_with_pad(self, x):
        """Custom resize operation that preserves aspect ratio with padding"""
        # Get current dimensions
        return torch.nn.functional.interpolate(
            x, self.target_size, mode='bilinear', align_corners=True
        )

    def forward(self, image, boxes):
        # Resize should be taken out of the forward function because it's not working in TensorRT
        # resized_image = self._resize_with_pad(image)
        # # Apply normalization
        backbone_out = self.model.forward_image((image - self.norm_mean) / self.norm_std)
        # backbone_out = self.model.forward_image(image)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in features["high_res_feats"]
        ]
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=None, boxes=boxes, masks=None
        )
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"][-1].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=boxes.shape[0] > 1,
            high_res_features=high_res_features,
        )
        return low_res_masks, iou_predictions

def preprocess_boxes(image, predictor, boxes=None):
    w, h = image.size
    orig_hw = [(h, w)]
    boxes = torch.as_tensor(boxes, dtype=torch.float, device='cuda')
    unnorm_box = predictor._transforms.transform_boxes(
        boxes, normalize=True, orig_hw=orig_hw[0]
    )  # Bx2x2
    return unnorm_box

def preprocess_inputs(image, predictor, boxes=None):
    w, h = image.size
    orig_hw = [(h, w)]
    input_image = predictor._transforms(image)
    input_image = input_image[None, ...].to('cuda')
    boxes = torch.as_tensor(boxes, dtype=torch.float, device='cuda')
    unnorm_box = predictor._transforms.transform_boxes(
        boxes, normalize=True, orig_hw=orig_hw[0]
    )  # Bx2x2
    return input_image, unnorm_box

def postprocess_masks(out, predictor, image):
    """Postprocess low-resolution masks and convert them for visualization."""
    orig_hw = (image.size[1], image.size[0])  # (height, width)
    masks = predictor._transforms.postprocess_masks(out[0], orig_hw)
    masks = (masks > 0.0).squeeze(0).cpu().numpy()
    scores = out[1].squeeze(0).cpu().numpy()
    return masks, scores


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()



checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
onnx_path = "assets/sam_model.onnx"

input_image = Image.open("./notebooks/images/truck.jpg").convert("RGB")
image = np.array(input_image).astype(np.float32) / 255.0
image = image.transpose(2, 0, 1)
image_tensor = torch.from_numpy(image).unsqueeze(0).cuda()
boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])
boxes_tensor = torch.from_numpy(boxes).cuda()
use_box = True
generate_onnx = True
with torch.no_grad():
    sam2_base = build_sam2(model_cfg, checkpoint)
    sam2_base.eval().cuda()
    predictor = SAM2ImagePredictor(sam2_base)

    encoder = predictor.model.eval().cuda()
    sam_model = SAM2FullModel(encoder)
    sam_model.eval().cuda()
    
    processed_boxes = preprocess_boxes(input_image, predictor, boxes)
    image_tensor = torch.nn.functional.interpolate(
        image_tensor, (1024, 1024), mode='bilinear', align_corners=True
    )
    torchtrt_inputs = (image_tensor, processed_boxes)
    # torchtrt_inputs = preprocess_inputs(input_image, predictor, boxes)
    input_names = ["input_image", "boxes"]
    if generate_onnx:
        outputs = sam_model(*torchtrt_inputs)
        masks, scores = postprocess_masks(outputs, predictor, input_image)
        plt.figure(figsize=(10, 10))
        plt.imshow(input_image)
        for mask in masks:
            show_mask(mask.squeeze(0), plt.gca(), random_color=True, borders=False)
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis('off')
        plt.show()
        # show_masks(input_image, masks, scores, point_coords=None, box_coords=boxes, input_labels=None, borders=False)

        print("Starting ONNX export...")
        # Try with a much lower opset version
        torch.onnx.export(
            sam_model,
            f=onnx_path,
            args=torchtrt_inputs,
            input_names=input_names,
            output_names=["low_res_masks", "iou_predictions"],
            do_constant_folding=True,
            dynamic_axes=None,
            export_params=True,
            opset_version=20,  # Try a much lower opset version
            training=torch.onnx.TrainingMode.EVAL,
        )
        print("ONNX export completed successfully")
        



################# TRT Export ##################
trt_cache_path = "/offboard/sam2/assets/trt_cache"
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,                     # Select GPU to execute
        "trt_engine_cache_enable": True,
        'trt_engine_cache_path': trt_cache_path,
        'trt_fp16_enable': True,              # Enable FP16 precision for faster inference  
        # 'trt_profile_opt_shapes': f"input_ids:1x{seq_len},attention_mask:1x{seq_len},position_ids:1x{seq_len},token_type_ids:1x{seq_len},text_token_mask:1x{seq_len}x{seq_len}",
        # 'trt_profile_min_shapes': "input_ids:1x1,attention_mask:1x1,position_ids:1x1,token_type_ids:1x1,text_token_mask:1x1x1",
        # 'trt_profile_max_shapes': "input_ids:1x256,attention_mask:1x256,position_ids:1x256,token_type_ids:1x256,text_token_mask:1x256x256",
        # 'trt_layer_norm_fp32_fallback': True, 
    }),
]
sess_opt = ort.SessionOptions()
# sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_opt.log_severity_level = 0
sess = ort.InferenceSession(onnx_path, providers=providers, sess_options=sess_opt)
io_binding = sess.io_binding()
device_type = "cuda"

binded_low_res_masks = torch.zeros_like(outputs[0])
binded_iou_predictions = torch.zeros_like(outputs[1])
io_binding.bind_input(name='input_image', device_type=device_type, device_id=0, element_type=np.float32, shape=torchtrt_inputs[0].shape, buffer_ptr=torchtrt_inputs[0].data_ptr())
io_binding.bind_input(name='boxes', device_type=device_type, device_id=0, element_type=np.float32, shape=torchtrt_inputs[1].shape, buffer_ptr=torchtrt_inputs[1].data_ptr())

io_binding.bind_output(name='low_res_masks', device_type=device_type, device_id=0, element_type=np.float32, shape=binded_low_res_masks.shape, buffer_ptr=binded_low_res_masks.data_ptr())
io_binding.bind_output(name='iou_predictions', device_type=device_type, device_id=0, element_type=np.float32, shape=binded_iou_predictions.shape, buffer_ptr=binded_iou_predictions.data_ptr())
with torch.no_grad():
    timer_pytorch = Timer(
        # The computation which will be run in a loop and timed.
        stmt="sam_model(*torchtrt_inputs)",
        setup="""
        """,
        
        globals={
            "sam_model": sam_model,
            "torchtrt_inputs": torchtrt_inputs
        },
        # Control the number of threads that PyTorch uses. (Default: 1)
        num_threads=1,
    )
    timer_onnx = Timer(
        # The computation which will be run in a loop and timed.
        stmt="sess.run_with_iobinding(io_binding)",
        # `setup` will be run before calling the measurement loop, and is used to
        # populate any state which is needed by `stmt`
        setup="""
        """,
        
        globals={
            "sess": sess,
            "io_binding": io_binding,
        },
        # Control the number of threads that PyTorch uses. (Default: 1)
        num_threads=1,
    )
    m_torch: Measurement = timer_pytorch.blocked_autorange(min_run_time=1)
    m_onnx: Measurement = timer_onnx.blocked_autorange(min_run_time=1)
print(m_torch)
print(m_onnx)
masks, scores = postprocess_masks((binded_low_res_masks, binded_iou_predictions), predictor, input_image)
plt.figure(figsize=(10, 10))
plt.imshow(input_image)
for mask in masks:
    show_mask(mask.squeeze(0), plt.gca(), random_color=True, borders=False)
for box in boxes:
    show_box(box, plt.gca())
plt.axis('off')
plt.show()
# outputs = sess.run_with_iobinding(io_binding)
# print(outputs)