# ------------------------------------------------------------------------
# Script to extract visual features from videos using DINO and CLIP models.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import argparse
import os
import torch

import moviepy.editor as mp
from transformers import AutoImageProcessor, AutoProcessor, Dinov2Model, Dinov2Config, CLIPModel
import numpy as np

def extract_dino_features(input_file, output_file):
    """
    Extracts DINO features from a video file and saves them to an output file.

    Args:
        input_file (str): The path to the input video file.
        output_file (str): The path to save the extracted features in npy format.
    """

    # Load the video file
    video = mp.VideoFileClip(input_file)

    # Create the DINO model
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    config = Dinov2Config(return_dict=True)
    model = Dinov2Model(config).from_pretrained("facebook/dinov2-large").to("cuda")
    
    # Extract framewise features using DINO
    features = []
    length_video = int(video.duration * video.fps)
    for i, frame in enumerate(video.iter_frames()):
        # Preprocess the frame
        frame = processor(images=frame, return_tensors="pt").to("cuda")
        # Pass the frame through the DINO model
        with torch.no_grad():
            feature = model(**frame).pooler_output.detach().cpu()

        if i % 100 == 0:
            print(str(i) + '/' + str(length_video))
        # Add the features to the list
        features.append(feature.squeeze().tolist())

    # Convert the features to a numpy array
    features = np.array(features)

    # Save the features to an output file in npy format
    np.save(output_file, features)
    
    print("Dino features extracted and saved to", output_file)


def extract_clip_features(input_file, output_file):
    """
    Extracts framewise features from a video using the CLIP model and saves them to an output file.

    Args:
        input_file (str): The path to the input video file.
        output_file (str): The path to save the extracted features in npy format.
    """
    
    # Load the video file
    video = mp.VideoFileClip(input_file)

    # Create the CLIP model
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
    
    # Extract framewise features using CLIP
    features = []
    length_video = int(video.duration * video.fps)
    for i, frame in enumerate(video.iter_frames()):
        # Preprocess the frame
        frame = processor(images=frame, return_tensors="pt").to("cuda")
        # Pass the frame through the CLIP model
        with torch.no_grad():
            feature = model.get_image_features(**frame).detach().cpu()

        if i % 100 == 0:
            print(str(i) + '/' + str(length_video))
        # Add the features to the list
        features.append(feature.squeeze().tolist())

    # Convert the features to a numpy array
    features = np.array(features)

    # Save the features to an output file in npy format
    np.save(output_file, features)
    print("CLIP features extracted and saved to", output_file)


def main(args):
    """
    Main function for extracting visual features from videos.
    
    This function extracts visual features from videos based on the specified dataset.
    It determines the number of subjects based on the dataset and iterates over each subject.
    For each subject, it extracts Dino features and clip features from the input video file.
    The extracted features are saved in separate output files.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    
    if args.dataset == 'actionsense':
        subjects = 9
    elif args.dataset == 'wear':
        subjects = 18
    elif args.dataset == 'wetlab':
        subjects = 22
    for i in range(subjects):
        input_file = os.path.join(args.video_dir, 'sbj_'+str(i)+'-12fps.mp4')
        if not os.path.exists(args.dino_output_dir):
            os.makedirs(args.dino_output_dir)
        if not os.path.exists(args.clip_output_dir):
            os.makedirs(args.clip_output_dir)
        dino_output_file = os.path.join(args.dino_output_dir, 'sbj_'+str(i))
        clip_output_file = os.path.join(args.clip_output_dir, 'sbj_'+str(i))
        extract_dino_features(input_file, dino_output_file)
        extract_clip_features(input_file, clip_output_file)

if __name__ == '__main__':
    # general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='actionsense', help='dataset name')
    parser.add_argument('--video_dir', type=str, default='data/actionsense/raw/ego/12fps/', help='input video directory')
    parser.add_argument('--clip_output_dir', type=str, default='data/actionsense/processed/clip_features/12fps_framewise', help='clip features output directory')
    parser.add_argument('--dino_output_dir', type=str, default='data/actionsense/processed/dino_features/12fps_framewise', help='dino features output directory')
    args = parser.parse_args()
    main(args)
    

