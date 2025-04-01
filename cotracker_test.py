import cv2
import torch
import numpy as np

from franka_utils import RealSense_Camera
camera = 'L515'
CAMERA_ID = 'f0211830'

realsense_camera = RealSense_Camera(type=camera, id=CAMERA_ID)
realsense_camera.prepare()
        
# Setup video capture
# cap = cv2.VideoCapture(0)  # 0 for webcam
device = 'cuda'
grid_size = 10  # Number of grid points to track

# Load model
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Initialize with first frames
frames = []
for i in range(cotracker.step * 2):  # Get enough frames to initialize
    # ret, frame = cap.read()
    _, rgbd_frame = realsense_camera.get_frame()
    if not rgbd_frame:
        break
    frame = rgbd_frame[:, :, :3]
    frames.append(frame)

# Convert to tensor format [B, T, C, H, W]
video_chunk = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)[None].float().to(device)
cotracker(video_chunk=video_chunk, is_first_step=True, grid_size=grid_size)

print("COTRACKER STEP: ", cotracker.step)

# Continue processing frames
while True:
    # frames = []
    for i in range(cotracker.step):  # Get enough frames for next chunk
        # ret, frame = cap.read()
        _, rgbd_frame = realsense_camera.get_frame()
        if not rgbd_frame:
            break
        frame = rgbd_frame[:, :, :3]
        frames.append(frame)
    
    if len(frames) < cotracker.step * 3:
        print("frames missing!")
    
    if len(frames) < cotracker.step * 2:
        print("not enough frames!")
        break
    
    frames = frames[-cotracker.step * 2:]
    
    # Convert to tensor format
    video_chunk = torch.tensor(np.array(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    
    # Get tracked positions for current frame
    pred_tracks, pred_visibility = cotracker(video_chunk=video_chunk)
    
    # Access the tracking results
    # pred_tracks has shape [B, T, N, 2] where N is number of points
    # pred_visibility has shape [B, T, N, 1]
    
    # Visualize if needed
    # show in realtime the last frame with tracked points
    frame = frames[-1]
    for track, visibility in zip(pred_tracks[0, -1], pred_visibility[0, -1]):
        if visibility.item() > 0.5:  # Only show points with high visibility
            x, y = int(track[0].item()), int(track[1].item())
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    
    # Display the frame
    cv2.imshow("Tracked Points", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break