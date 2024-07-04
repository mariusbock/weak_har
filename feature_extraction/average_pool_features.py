# ------------------------------------------------------------------------
# Script to perform average pooling on frame-wise features.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------

import argparse
import numpy as np
import glob

def main(args):
    """
    Main method for performing average pooling on features.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    files = sorted(glob.glob(f'{args.input_dir}/*.npy'))
    for feat in files:
        features = np.load(feat)
        print(feat, features.shape[0])
        fps = 12
        sbj_id = feat.split('/')[-1].split('.')[0].split('_')[1]
        # Calculate the number of records to average pool
        pool_size = int(args.window_s * fps)
        overlap = int(args.overlap_s * fps)
        stride = pool_size - overlap

        # Calculate the number of pools
        num_pools = (features.shape[0] - overlap) // (pool_size - overlap)
        print('sbj_' + sbj_id, num_pools)
        # Initialize an empty array to store the pooled data
        pooled_data = np.empty((num_pools, features.shape[1]))

        # Perform average pooling
        for i in range(num_pools):
            start = i * (pool_size - overlap)
            end = start + pool_size
            pooled_data[i] = np.mean(features[start:end], axis=0)

        # Assign the pooled data to flow_data
        pooled_features = pooled_data
        np.save(f'data/{args.dataset}/processed/{args.feature_type}_features/{pool_size}_frames_{stride}_stride_avg_pool/' + 'sbj_' + sbj_id + '.npy', pooled_features)
        


if __name__ == '__main__':
    # general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wetlab', help='dataset name')
    parser.add_argument('--feature_type', type=str, default='clip', help='feature type')
    parser.add_argument('--input_dir', type=str, default='data/wetlab/processed/clip_features/12fps_framewise', help='input feature directory')
    parser.add_argument('--window_s', type=str, default=2, help='window size (in seconds)')
    parser.add_argument('--overlap_s', type=str, default=1, help='window overlap (in seconds)')
    args = parser.parse_args()
    main(args) 
    
