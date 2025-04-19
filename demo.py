import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sam2.build_sam import build_sam2_video_predictor

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
video_dir = "./notebooks/videos/bedroom"
# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=video_dir)
    
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    # Create a figure to collect the box selection
    click_fig, click_ax = plt.subplots(figsize=(12, 8))
    click_ax.set_title("Draw a box around the object (click and drag, then press Enter)")
    frame_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    click_ax.imshow(Image.open(frame_path))
    
    box = []
    rect = None
    start_point = None
    
    def onclick(event):
        global start_point, rect
        if event.xdata is not None and event.ydata is not None:
            start_point = (event.xdata, event.ydata)
    
    def onrelease(event):
        global start_point, rect, box
        if start_point and event.xdata is not None and event.ydata is not None:
            end_point = (event.xdata, event.ydata)
            # Create box coordinates [x1, y1, x2, y2]
            x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
            x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
            box = [[x1, y1], [x2, y2]]
            
            # Remove previous rectangle if it exists
            if rect:
                rect.remove()
            
            # Draw new rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                 fill=False, edgecolor='green', linewidth=2)
            click_ax.add_patch(rect)
            click_fig.canvas.draw_idle()
            start_point = None
    
    def on_key(event):
        if event.key == 'enter':
            plt.close(click_fig)
    
    click_fig.canvas.mpl_connect('button_press_event', onclick)
    click_fig.canvas.mpl_connect('button_release_event', onrelease)
    click_fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
    
    # Use the box selection or default to a predefined box
    if not box:
        print("No box selected. Using default box.")
        box = [[210, 220], [250, 350]]
    
    # Pass the box to the predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        box=box,
    )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # render the segmentation results every few frames
    vis_frame_stride = 30
    plt.close("all")

    # Replace the final visualization loop with interactive keyboard navigation
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    current_frame_idx = 0

    def update_frame(frame_idx):
        ax.clear()
        ax.set_title(f"Frame {frame_idx}")
        frame_path = os.path.join(video_dir, frame_names[frame_idx])
        ax.imshow(Image.open(frame_path))
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                show_mask(out_mask, ax, obj_id=out_obj_id)
        fig.canvas.draw_idle()

    def on_key_press(event):
        global current_frame_idx
        if event.key == 'right' or event.key == 'd':
            current_frame_idx = min(current_frame_idx + 1, len(frame_names) - 1)
        elif event.key == 'left' or event.key == 'a':
            current_frame_idx = max(current_frame_idx - 1, 0)
        elif event.key == 'up' or event.key == 'w':
            current_frame_idx = min(current_frame_idx + 10, len(frame_names) - 1)
        elif event.key == 'down' or event.key == 's':
            current_frame_idx = max(current_frame_idx - 10, 0)
        update_frame(current_frame_idx)

    # Connect the key press event to the handler
    fig.canvas.mpl_connect('key_press_event', on_key_press)

    # Show initial frame
    update_frame(current_frame_idx)
    plt.tight_layout()
    plt.show()