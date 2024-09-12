from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
import os
import uuid

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "path/to/sam2_hiera_large.pt"
CONFIG = "sam2_hiera_l.yaml"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Initialize SAM2 model
sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'video_id': filename})
    return jsonify({'error': 'Invalid file type'})

@app.route('/select_objects', methods=['POST'])
def select_objects():
    data = request.json
    video_id = data.get('video_id')
    objects = data.get('objects', [])
    
    if not video_id or not objects:
        return jsonify({'error': 'Missing video_id or objects data'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found'})
    
    inference_state = sam2_model.init_state(video_path=video_path)
    
    for obj_id, obj in enumerate(objects, start=1):
        points = np.array(obj['points'], dtype=np.float32)
        labels = np.ones(len(points))
        sam2_model.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=labels
        )
    
    # Save the inference state for later use
    state_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{video_id}_state")
    torch.save(inference_state, state_path)
    
    return jsonify({'message': 'Objects selected successfully', 'video_id': video_id})

@app.route('/track_objects', methods=['POST'])
def track_objects():
    data = request.json
    video_id = data.get('video_id')
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
    state_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{video_id}_state")
    
    if not os.path.exists(video_path) or not os.path.exists(state_path):
        return jsonify({'error': 'Video or state not found'})
    
    inference_state = torch.load(state_path)
    
    output_folder = os.path.join(app.config['PROCESSED_FOLDER'], str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    output_video = os.path.join(output_folder, 'tracked.mp4')
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
        ret, frame = video.read()
        if not ret:
            break
        
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        
        for obj_id, mask in zip(object_ids, masks):
            frame[mask] = frame[mask] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
        
        out.write(frame)
    
    video.release()
    out.release()
    
    return jsonify({'message': 'Objects tracked successfully', 'output_video': output_video})

@app.route('/apply_effect', methods=['POST'])
def apply_effect():
    data = request.json
    video_id = data.get('video_id')
    effect = data.get('effect', 'overlay')
    
    if not video_id:
        return jsonify({'error': 'Missing video_id'})
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_id)
    state_path = os.path.join(app.config['PROCESSED_FOLDER'], f"{video_id}_state")
    
    if not os.path.exists(video_path) or not os.path.exists(state_path):
        return jsonify({'error': 'Video or state not found'})
    
    inference_state = torch.load(state_path)
    
    output_folder = os.path.join(app.config['PROCESSED_FOLDER'], str(uuid.uuid4()))
    os.makedirs(output_folder, exist_ok=True)
    output_video = os.path.join(output_folder, f'{effect}.mp4')
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
        ret, frame = video.read()
        if not ret:
            break
        
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        
        for obj_id, mask in zip(object_ids, masks):
            if effect == 'overlay':
                frame[mask] = frame[mask] * 0.5 + np.array([0, 0, 255], dtype=np.uint8) * 0.5
            elif effect == 'erase':
                frame[mask] = 0
            elif effect == 'pixelate':
                pixelated = cv2.resize(frame[mask], (32, 32), interpolation=cv2.INTER_LINEAR)
                frame[mask] = cv2.resize(pixelated, (mask.sum(), 1), interpolation=cv2.INTER_NEAREST).reshape(frame[mask].shape)
        
        out.write(frame)
    
    video.release()
    out.release()
    
    return jsonify({'message': f'Effect {effect} applied successfully', 'output_video': output_video})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
