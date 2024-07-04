# ------------------------------------------------------------------------
# Test script to assess the annotation of weak labels
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
from copy import deepcopy
import warnings

from clustering import apply_gmm, create_correlation_matrix, sample_cluster_feat
from utils.data_utils import unwindow_inertial_data
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import datetime
import seaborn as sns
import json
import os
from pprint import pprint
import sys
import time

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import torch

from utils.torch_utils import InertialDataset, fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# set the precision of numpy arrays to 2 decimal places
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

import numpy as np
from scipy.spatial.distance import cdist

class ClusterSimilarityMatrix():
    """
    Class for computing the similarity matrix of cluster labels.
    """

    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        """
        Fits the similarity matrix with the given cluster labels.

        Args:
            y_clusters (array-like): The cluster labels.

        Returns:
            self: Returns the fitted instance of ClusterSimilarityMatrix.
        """
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        """
        Converts the cluster labels into a binary similarity matrix.

        Args:
            y_clusters (array-like): The cluster labels.

        Returns:
            array-like: The binary similarity matrix.
        """
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)
        
    
def main(args):
    """
    Main function that performs the annotation pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    # load config
    args.config = 'configs/{}/annotation_pipeline_{}sec.yaml'.format(args.dataset, args.clip_length)
    args.output_folder = 'output_preds/{}/{}'.format(args.dataset, args.features)  
    config = load_config(args.config)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['init_rand_seed'] = args.seed
    # set experiment parameters
    config['train_cfg']['supervision_type'] = args.supervision
    config['train_cfg']['training_type'] = args.training_type
    config['clustering']['nb_clusters'] = args.cluster
    config['clustering']['nb_samples'] = args.samples
    config['clustering']['sampling_strategy'] = args.sampling_strategy
    config['train_cfg']['loss_type'] = args.loss
    config['train_cfg']['label_smoothing'] = args.label_smoothing
    config['train_cfg']['tau'] = args.tau

    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts))
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # set random seed for reproducibility
    fix_random_seed(config['init_rand_seed'])

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
        
    all_v_pred = np.array([])
    all_v_gt = np.array([])
    records = 0
    # iterate over all subjects
    for i, anno_split in enumerate(config['anno_json']):
        # load labels and train/val subject ids
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if config['has_null'] == True:
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        
        train_names = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
        
        # load train and val inertial and embedding data
        sens_data = np.empty((0, config['dataset']['input_dim'] + 2))
        conv3d_emb = np.empty((0, config['dataset']['conv3d_dim']))
        raft_emb = np.empty((0, config['dataset']['raft_dim']))
        dino_emb = np.empty((0, config['dataset']['dino_dim']))
        clip_emb = np.empty((0, config['dataset']['clip_dim']))
        
        for sbj in train_names:
            ts_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'], sbj + '.csv'), index_col=False, low_memory=False).replace({"label": config['label_dict']}).fillna(0).to_numpy()
            sens_data = np.append(sens_data, ts_data, axis=0)
            c3d_data = np.load(os.path.join(config['dataset']['conv3d_folder'], sbj + '.npy')).astype(np.float32)
            r_data = np.load(os.path.join(config['dataset']['raft_folder'], sbj + '.npy')).astype(np.float32)
            d_data = np.load(os.path.join(config['dataset']['dino_folder'], sbj + '.npy')).astype(np.float32)
            c_data = np.load(os.path.join(config['dataset']['clip_folder'], sbj + '.npy')).astype(np.float32)
            conv3d_emb = np.append(conv3d_emb, c3d_data, axis=0)
            raft_emb = np.append(raft_emb, r_data, axis=0)
            dino_emb = np.append(dino_emb, d_data, axis=0)
            clip_emb = np.append(clip_emb, c_data, axis=0)
        
        print(sbj, clip_emb.shape, raft_emb.shape)
        dino_emb = dino_emb[:len(raft_emb)]
        clip_emb = clip_emb[:len(raft_emb)]
        if args.features == 'dino':
            emb_data = dino_emb
        elif args.features == 'clip':
            emb_data = clip_emb
        elif args.features == 'raft':
            emb_data = raft_emb
        elif args.features == 'conv3d':
            emb_data = conv3d_emb
        elif args.features == 'i3d':
            emb_data = np.concatenate((conv3d_emb, raft_emb), axis=1)
        elif args.features == 'dino+raft':
            emb_data = np.concatenate((raft_emb, dino_emb), axis=1)
        elif args.features == 'clip+raft':
            emb_data = np.concatenate((clip_emb, raft_emb), axis=1)

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split

        # define inertial datasets
        train_dataset = InertialDataset(sens_data, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'], classes=config['dataset']['nb_classes'])

        ######## CLUSTERING #######
        v_cluster_labels, v_cluster_dist, _ = apply_gmm(emb_data, config['clustering']['nb_clusters'], seed=config['init_rand_seed'])
        v_cluster_labels, v_cluster_dist = v_cluster_labels[:len(train_dataset.labels)], v_cluster_dist[:len(train_dataset.labels)]

        # sample instances from clusters
        v_mask = sample_cluster_feat(v_cluster_labels, v_cluster_dist, config['clustering']['nb_samples'], config['clustering']['nb_clusters'], config['clustering']['sampling_strategy'])

        # create correlation matrix based on sampled instances
        corr_matrix = create_correlation_matrix(train_dataset.labels[v_mask], v_cluster_labels[v_mask], config['clustering']['nb_clusters'], config['dataset']['nb_classes'])
                    
        #### CONVERT CLUSTERING TO PREDICTIONS
        v_preds = np.zeros_like(train_dataset.labels)
        v_dist = np.zeros_like(train_dataset.labels)
        max_v_corr_matrix = np.argmax(corr_matrix, axis=1)
        for k, v_clu_label in enumerate(v_cluster_labels):            
            v_preds[k, max_v_corr_matrix[v_clu_label]] = 1
            v_dist[k, max_v_corr_matrix[v_clu_label]] = v_cluster_dist[k]
        if args.thresholded:
            t_mask = v_cluster_dist < args.threshold
        train_dataset.labels = v_preds
        train_dataset.weights = v_dist

        labels, weights = np.argmax(train_dataset.labels, axis=1), np.max(train_dataset.weights, axis=1)
        
        if args.thresholded:
            labels[~t_mask], weights[~t_mask] = -1, -1
        train_dataset.labels, train_dataset.weights = labels, weights

        orig_sens_data = deepcopy(sens_data)
        cl, _ = unwindow_inertial_data(sens_data, train_dataset.ids, train_dataset.labels, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'])
        weights, _ = unwindow_inertial_data(sens_data, train_dataset.ids, train_dataset.weights, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'])
        sens_data[:, -1] = cl
        if args.thresholded:
            orig_sens_data = orig_sens_data[sens_data[:, -1] != -1]
            sens_data = sens_data[sens_data[:, -1] != -1]
            weights = weights[weights != -1]
        records += len(sens_data)
        
        v_preds, v_gt = sens_data[:, -1], orig_sens_data[:, -1]
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        np.save(os.path.join(args.output_folder, f'sbj_{int(i)}_pred.npy'), v_preds)
        np.save(os.path.join(args.output_folder, f'sbj_{int(i)}_gt.npy'), v_gt)

        # Distribution of values in v_preds
        unique_values, value_counts = np.unique(v_preds, return_counts=True)
        distribution = dict(zip(unique_values, value_counts))
        print("Distribution of values in v_preds:")
        for value, count in distribution.items():
            print(f"{value}: {count} occurrences")

        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=0)
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=0)
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=0)

        # print to terminal
        block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = ''
        block2  += '\t\tAcc {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_acc * 100)))
        block2  += ' Prec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_prec * 100)))
        block2  += ' Rec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_rec * 100)))
        block2  += ' F1 {:>4.2f} (%)\n'.format(np.mean(np.nan_to_num(v_f1 * 100)))

        print('\n'.join([block1, block2]))
                                
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\t\tAcc {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_acc * 100)))
    block2  += ' Prec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_prec * 100)))
    block2  += ' Rec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_rec * 100)))
    block2  += ' F1 {:>4.2f} (%)\n'.format(np.mean(np.nan_to_num(v_f1 * 100)))
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    conf_mat = np.around(conf_mat, 2)
    conf_mat[conf_mat == 0] = np.nan
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                'fontsize': 16,})
    ax.set_title('Confusion Matrix')
    _.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    print(records)
    print("ALL FINISHED")

if __name__ == '__main__':
    # general arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='clip+raft', help='conv3d, raft, clip, dino, clip+raft, dino+raft, i3d')
    parser.add_argument('--dataset', type=str, default='actionsense', help='wear, wetlab, actionsense')
    parser.add_argument('--clip_length', default=4, type=int, help='sliding window clip length in seconds')
    parser.add_argument('--seed', default=1, type=int)       
    # experiment arguments
    parser.add_argument('--supervision', default='weak_labeling', type=str)
    parser.add_argument('--training_type', default='normal', type=str)
    parser.add_argument('--cluster', default=100, type=int)
    parser.add_argument('--samples', default=1, type=int)
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--thresholded', default=False, action='store_true')
    parser.add_argument('--threshold', default=2.0, type=float)
    parser.add_argument('--loss', default='phgce', type=str)
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--sampling_strategy', default='distance', type=str)
    parser.add_argument('--tau', default=10, type=float)
    args = parser.parse_args()
    main(args)  

