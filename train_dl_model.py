# ------------------------------------------------------------------------
# Main script used to commence deep learning experiments
# ------------------------------------------------------------------------
# Author: anonymised
# E-Mail: anonymised
# ------------------------------------------------------------------------
import warnings

import torch
from torch import nn
from utils.data_utils import unwindow_inertial_data

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time 
from copy import deepcopy

import pandas as pd
import numpy as np
import neptune
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from models.inertial.train import run_inertial_network
from utils.torch_utils import InertialDataset, fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from clustering import apply_gmm, create_correlation_matrix, sample_cluster_feat


def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="",
        api_token=""
        )
    else:
        run = None
    config = load_config(args.config)
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['init_rand_seed'] = args.seed
    # set experiment parameters
    config['train_cfg']['supervision_type'] = args.supervision
    config['train_cfg']['training_type'] = args.training_type
    config['clustering']['used_features'] = args.features
    config['clustering']['nb_clusters'] = args.cluster
    config['clustering']['nb_samples'] = args.samples
    config['clustering']['sampling_strategy'] = args.sampling_strategy
    config['train_cfg']['loss_type'] = args.loss
    config['train_cfg']['label_smoothing'] = args.label_smoothing
    config['train_cfg']['tau'] = args.tau

    if args.neptune:
        run_id = run["sys/id"].fetch()
    else:
        run_id = args.run_id
    
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    log_dir = os.path.join('logs', config['name'], str(ts) + '_' + run_id)
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
    
    if args.neptune:
        run['name'] = config['name']
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))
        run['train_cfg'] = args.supervision
        if args.supervision != 'full_supervision':
            run['features'] = args.features
            run['cluster'] = args.cluster
            run['samples'] = args.samples
            run['weighted'] = args.weighted
            run['training_type'] = args.training_type
        run['thresholded'] = args.thresholded
        if args.thresholded:
            run['threshold'] = args.threshold
        run['loss'] = args.loss
        run['label_smoothing'] = args.label_smoothing
        run['sampling_strategy'] = args.sampling_strategy
        if args.loss == 'phce' or  args.loss == 'phgce':
            run['tau'] = args.tau
 
    rng_generator = fix_random_seed(config['init_rand_seed'])    

    all_v_pred = np.array([])
    all_v_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))
    
    with open(config['anno_json'][0]) as f:
            file = json.load(f)
            
    anno_file = file['database']
    if config['has_null'] == True:
        config['labels'] = ['null'] + list(file['label_dict'])
    else:
        config['labels'] = list(file['label_dict'])
    
    config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
    all_sbjs = sorted([x for x in anno_file], key=lambda x: int(x.split('_')[-1]))

    sens_data = np.empty((0, config['dataset']['input_dim'] + 2))
    for sbj in all_sbjs:
        ts_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'], sbj + '.csv'), index_col=False, low_memory=False).replace({"label": config['label_dict']}).fillna(0).to_numpy()
        sens_data = np.append(sens_data, ts_data, axis=0)
    
    conv3d_emb = np.empty((0, config['dataset']['conv3d_dim']))
    raft_emb = np.empty((0, config['dataset']['raft_dim']))
    dino_emb = np.empty((0, config['dataset']['dino_dim']))
    clip_emb = np.empty((0, config['dataset']['clip_dim']))
    if config['train_cfg']['supervision_type'] == "weak_labelling":
        for sbj in all_sbjs:
            c3d_data = np.load(os.path.join(config['dataset']['conv3d_folder'], sbj + '.npy')).astype(np.float32)
            r_data = np.load(os.path.join(config['dataset']['raft_folder'], sbj + '.npy')).astype(np.float32)
            d_data = np.load(os.path.join(config['dataset']['dino_folder'], sbj + '.npy')).astype(np.float32)
            c_data = np.load(os.path.join(config['dataset']['clip_folder'], sbj + '.npy')).astype(np.float32)
            conv3d_emb = np.append(conv3d_emb, c3d_data, axis=0)
            raft_emb = np.append(raft_emb, r_data, axis=0)
            dino_emb = np.append(dino_emb, d_data, axis=0)
            clip_emb = np.append(clip_emb, c_data, axis=0)
        
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

    all_dataset = InertialDataset(sens_data, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'], classes=config['dataset']['nb_classes'])
    #print(all_dataset.labels.shape)
    ######## CLUSTERING #######
    if config['train_cfg']['supervision_type'] == "weak_labelling":
        sampled_mask = np.array([], dtype=bool)
        threshold_mask = np.array([], dtype=bool)
        for sbj_id in np.unique(all_dataset.ids):
            curr_emb, curr_labels = emb_data[all_dataset.ids == sbj_id, :], all_dataset.labels[all_dataset.ids == sbj_id]    
            #### APPLY CLUSTERING
            #curr_emb, _ = reduce_dimensionality(curr_emb, 'pca')
            v_cluster_labels, v_cluster_dist, _ = apply_gmm(curr_emb, config['clustering']['nb_clusters'], seed=config['init_rand_seed'])

            # sample instances from clusters
            v_mask = sample_cluster_feat(v_cluster_labels, v_cluster_dist, config['clustering']['nb_samples'], config['clustering']['nb_clusters'], config['clustering']['sampling_strategy'])
            sampled_mask = np.concatenate((sampled_mask, v_mask))

            # create correlation matrix based on sampled instances
            corr_matrix = create_correlation_matrix(curr_labels[v_mask], v_cluster_labels[v_mask], config['clustering']['nb_clusters'], config['dataset']['nb_classes'])
                    
            #### CONVERT CLUSTERING TO PREDICTIONS
            if config['train_cfg']['supervision_type'] == "weak_labelling":
                v_preds = np.zeros_like(curr_labels)
                v_dist = np.zeros_like(curr_labels)
                max_v_corr_matrix = np.argmax(corr_matrix, axis=1)
                for k, v_clu_label in enumerate(v_cluster_labels):            
                    v_preds[k, max_v_corr_matrix[v_clu_label]] = 1
                    v_dist[k, max_v_corr_matrix[v_clu_label]] = v_cluster_dist[k]
                if args.thresholded:
                    v_mask = v_cluster_dist < args.threshold
                    threshold_mask = np.concatenate((threshold_mask, v_mask))
                all_dataset.labels[all_dataset.ids == sbj_id] = v_preds
                all_dataset.weights[all_dataset.ids == sbj_id] = v_dist
            
        labels, weights = np.argmax(all_dataset.labels, axis=1), np.max(all_dataset.weights, axis=1)
        
        if args.training_type == "few_shot":
            labels[~sampled_mask], weights[~sampled_mask] = -1, -1            
        
        if args.thresholded:
            labels[~threshold_mask], weights[~threshold_mask] = -1, -1
        all_dataset.labels, all_dataset.weights = labels, weights

        orig_sens_data = deepcopy(sens_data)
        cl, _ = unwindow_inertial_data(sens_data, all_dataset.ids, all_dataset.labels, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'])
        weights, _ = unwindow_inertial_data(sens_data, all_dataset.ids, all_dataset.weights, config['dataset']['c_window_size'], config['dataset']['c_window_overlap'])
        sens_data[:, -1] = cl
        if args.training_type == 'few_shot' or args.thresholded:
            orig_sens_data = orig_sens_data[sens_data[:, -1] != -1]
            sens_data = sens_data[sens_data[:, -1] != -1]
            weights = weights[weights != -1]
    
    # Define complete dataset 
    if args.weighted:
        all_dataset = InertialDataset(sens_data, config['dataset']['t_window_size'], config['dataset']['t_window_overlap'], weights, classes=config['dataset']['nb_classes'])
    else:
        all_dataset = InertialDataset(sens_data, config['dataset']['t_window_size'], config['dataset']['t_window_overlap'], classes=config['dataset']['nb_classes'])
        
    if config['train_cfg']['supervision_type'] == "weak_labelling":
        for s in np.unique(orig_sens_data[:, 0]):
            fil_pred, fil_gt = sens_data[sens_data[:, 0] == s][:, -1], orig_sens_data[orig_sens_data[:, 0] == s][:, -1]
            test_mat = confusion_matrix(fil_pred, fil_gt, normalize='true', labels=range(len(config['labels'])))
            labeling_accuracy = np.nan_to_num(test_mat.diagonal()/test_mat.sum(axis=1))
            print('Label Accuracy Subject {}:\t {:.2f}'.format('sbj_' + str(int(s)), np.mean(labeling_accuracy) * 100))
            if run is not None:
                run['sbj_' + str(int(s)) + '_labeling_accuracy'] = np.nanmean(labeling_accuracy)
                run['final_labeling_accuracy'] = np.nanmean(labeling_accuracy)
        test_mat = confusion_matrix(sens_data[:, -1], orig_sens_data[:, -1], normalize='true', labels=range(len(config['labels'])))
        labeling_accuracy = np.nan_to_num(test_mat.diagonal()/test_mat.sum(axis=1))
        print('AVG. LABEL ACCURACY:\t {:.2f}'.format(np.mean(labeling_accuracy) * 100))
        if run is not None:
            run['final_labeling_accuracy'] = np.mean(labeling_accuracy)
    
    ######## TRAIN CLASSIFIER #######
    for i, anno_split in enumerate(config['anno_json']):
        # load labels and train/val subject ids
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        train_sbjs = sorted([int(x.split('_')[-1]) for x in anno_file if anno_file[x]['subset'] == 'Training'])
        val_sbjs = sorted([x for x in anno_file if anno_file[x]['subset'] == 'Validation'], key=lambda x: int(x.split('_')[-1]))
        
        # load train and val inertial data
        val_sens_data = np.empty((0, config['dataset']['input_dim'] + 2))
        for v_sbj in val_sbjs:
            vs_data = pd.read_csv(os.path.join(config['dataset']['sens_folder'], v_sbj + '.csv'), index_col=False, low_memory=False).replace({"label": config['label_dict']}).fillna(0).to_numpy()
            val_sens_data = np.append(val_sens_data, vs_data, axis=0)
        mask = np.isin(sens_data[:, 0], train_sbjs)
        train_dataset = InertialDataset(sens_data[mask], config['dataset']['t_window_size'], config['dataset']['t_window_overlap'], classes=config['dataset']['nb_classes'])
       
        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        config['train_cfg']['loss_type'] = args.loss
        t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_dataset, val_sens_data, config, log_dir, args.ckpt_freq, rng_generator, run)
            
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=0, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=0, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=0, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.mean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.mean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(np.nan_to_num(v_mAP) * 100))
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], np.nan_to_num(v_mAP)):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_acc * 100)))
        block5  += ' Prec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_prec * 100)))
        block5  += ' Rec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_rec * 100)))
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.mean(np.nan_to_num(v_f1 * 100)))

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=0, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(np.nan_to_num(all_v_mAP)) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], np.nan_to_num(all_v_mAP.T)):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.mean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_acc)) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_prec)) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_rec)) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_f1)) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.mean(np.nan_to_num(all_v_mAP))
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], np.nan_to_num(all_v_mAP.T)):
            run['final_mAP@' + str(tiou)] = np.mean(tiou_mAP)
        run['final_accuracy'] = np.mean(np.nan_to_num(v_acc))
        run['final_precision'] = np.mean(np.nan_to_num(v_prec))
        run['final_recall'] = np.mean(np.nan_to_num(v_rec))
        run['final_f1'] = np.mean(np.nan_to_num(v_f1))

    print("ALL FINISHED")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # run arguments
    parser.add_argument('--config', default='configs/wear/tinyhar.yaml')
    parser.add_argument('--run_id', default='', type=str)
    parser.add_argument('--eval_type', default='loso')
    parser.add_argument('--neptune', default=False, action='store_true')
    parser.add_argument('--seed', default=42, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    # experiment arguments
    parser.add_argument('--supervision', default='full_supervision', type=str, choices=['full_supervision', 'weak_labelling'])
    parser.add_argument('--training_type', default='normal', type=str, choices=['normal', 'few_shot'])
    parser.add_argument('--features', default='clip+raft', type=str, choices=['clip', 'dino', 'raft', 'conv3d', 'i3d', 'dino+raft', 'clip+raft'])
    parser.add_argument('--cluster', default=19, type=int, help='Number of clusters')
    parser.add_argument('--samples', default=1, type=int, help='Number of samples per cluster')
    parser.add_argument('--weighted', default=False, action='store_true', help='Use weighted loss (based on distances)')
    parser.add_argument('--thresholded', default=False, action='store_true', help='Use thresholded training')
    parser.add_argument('--threshold', default=4.0, type=float, help='Threshold for thresholded training')
    parser.add_argument('--loss', default='ce', type=str, choices=['phgce', 'phce', 'ce'])
    parser.add_argument('--label_smoothing', default=0.0, type=float, help='Label smoothing factor')
    parser.add_argument('--sampling_strategy', default='distance', type=str, choices=['distance', 'random'])
    parser.add_argument('--tau', default=10, type=float, help='Tau for PHCE/PHGCE loss')
    args = parser.parse_args()
    main(args)  

