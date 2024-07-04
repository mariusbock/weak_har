# ------------------------------------------------------------------------
# Methods used for training inertial-based models
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import os
import time
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from utils.data_utils import convert_samples_to_segments, unwindow_inertial_data
from utils.torch_utils import PHuberCrossEntropy, PHuberGeneralizedCrossEntropy, init_weights, save_checkpoint, worker_init_reset_seed, InertialDataset
from utils.os_utils import mkdir_if_missing
from models.inertial.ShallowDeepConvLSTM import ShallowDeepConvLSTM
from models.inertial.TinyHAR import TinyHAR
from utils.metrics import ANETdetection


def run_inertial_network(train_dataset, val_sens_data, cfg, ckpt_folder, ckpt_freq, rng_generator, run, start_epoch=0):
    """
    Runs the inertial network training and validation process.

    Args:
        train_dataset (InertialDataset): The training dataset.
        val_sens_data (numpy.ndarray): The validation sensor data.
        cfg (dict): The configuration dictionary.
        ckpt_folder (str): The folder to save checkpoints.
        ckpt_freq (int): The frequency of saving checkpoints.
        rng_generator (torch.Generator): The random number generator.
        run (dict): The dictionary to store the training run information.
        start_epoch (int, optional): The starting epoch. Defaults to 0.

    Returns:
        tuple: A tuple containing the training losses, validation losses, validation mAP, validation predictions, and validation ground truth.
    """    
    split_name = cfg['dataset']['json_anno'].split('/')[-1].split('.')[0]
    test_dataset = InertialDataset(val_sens_data, cfg['dataset']['t_window_size'], cfg['dataset']['t_window_overlap'], cfg['dataset']['nb_classes'])

    # define criterion 
    if cfg['train_cfg']['loss_type'] == 'ce':
        criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=cfg['train_cfg']['label_smoothing'])
    elif cfg['train_cfg']['loss_type'] == 'phce':
        criterion = PHuberCrossEntropy(tau=cfg['train_cfg']['tau'], label_smoothing=cfg['train_cfg']['label_smoothing'])
    elif cfg['train_cfg']['loss_type'] == 'phgce':
        criterion = PHuberGeneralizedCrossEntropy(tau=cfg['train_cfg']['tau'], label_smoothing=cfg['train_cfg']['label_smoothing'])
    val_criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=cfg['train_cfg']['label_smoothing'])

    # use weighted loss if selected
    if cfg['train_cfg']['weighted_loss']:
        class_weights = torch.from_numpy(np.ones(cfg['dataset']['nb_classes'])).float()
        activities, a_counts = np.unique(train_dataset.labels, return_counts=True)
        for i_a, a in zip(activities, a_counts):
            class_weights[int(i_a)] = a_counts.sum() / (cfg['dataset']['nb_classes'] * a)
        criterion.weight = torch.tensor(class_weights).float().to(cfg['device'])
    
    ######## TRAINING #######
    # define dataloaders
    train_loader = DataLoader(train_dataset, cfg['loader']['batch_size'], shuffle=cfg['loader']['shuffle'], num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    test_loader = DataLoader(test_dataset, cfg['loader']['batch_size'], shuffle=False, num_workers=4, worker_init_fn=worker_init_reset_seed, generator=rng_generator, persistent_workers=True)
    
    # define network
    if cfg['name'] == 'dim_deepconvlstm':
        net = ShallowDeepConvLSTM(
            train_dataset.channels, train_dataset.classes, train_dataset.window_size,
            cfg['model']['conv_kernels'], cfg['model']['conv_kernel_size'], 
            cfg['model']['lstm_units'], cfg['model']['lstm_layers'], cfg['model']['dropout']
            )
        print("Number of learnable parameters for DeepConvLSTM: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    elif cfg['name'] == 'tinyhar':
        net = TinyHAR((cfg['loader']['batch_size'], 1, train_dataset.features.shape[1], train_dataset.channels), train_dataset.classes, 
                    cfg['model']['conv_kernels'], cfg['model']['conv_layers'], cfg['model']['conv_kernel_size'], dropout=cfg['model']['dropout']
                    )
        print("Number of learnable parameters for TinyHAR: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    net = init_weights(net, cfg['train_cfg']['weight_init'])

    # define optimizer
    opt = torch.optim.Adam(net.parameters(), lr=cfg['train_cfg']['lr'], weight_decay=cfg['train_cfg']['weight_decay'])
    # use lr schedule if selected
    if cfg['train_cfg']['lr_step'] > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg['train_cfg']['lr_step'], gamma=cfg['train_cfg']['lr_decay'])

    net.to(cfg['device'])
    for epoch in range(start_epoch, start_epoch + cfg['train_cfg']['epochs']):
        start_time = time.time()
        # training
        net, t_losses, _, _ = train_one_epoch(train_loader, net, opt, criterion, cfg['device'])
    
        print("--- %s seconds ---" % (time.time() - start_time))
        # save ckpt once in a while
        if (((ckpt_freq > 0) and ((epoch + 1) % ckpt_freq == 0))):
            save_states = { 
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': opt.state_dict(),
            }

            file_name = 'epoch_{:03d}_{}.pth.tar'.format(epoch + 1, split_name)
            save_checkpoint(save_states, False, file_folder=os.path.join(ckpt_folder, 'ckpts'), file_name=file_name)

        # validation
        v_losses, v_preds, v_gt = validate_one_epoch(test_loader, net, val_criterion, cfg['device'])
        
        if cfg['train_cfg']['lr_step'] > 0:
            scheduler.step()
        
        # use mAP calculation as in ActionFormer
        det_eval = ANETdetection(cfg['dataset']['json_anno'], 'validation', tiou_thresholds = cfg['dataset']['tiou_thresholds'])
        # undwindow inertial data (sample-wise structure instead of windowed) 
        v_preds, v_gt = unwindow_inertial_data(val_sens_data, test_dataset.ids, v_preds, cfg['dataset']['t_window_size'], cfg['dataset']['t_window_overlap'])
        # convert to samples (for mAP calculation)
        v_segments = convert_samples_to_segments(val_sens_data[:, 0], v_preds, cfg['dataset']['sampling_rate'])

        if epoch == (start_epoch + cfg['train_cfg']['epochs']) - 1:
            # save raw results (for later postprocessing)
            v_results = pd.DataFrame({
                'video_id' : v_segments['video-id'],
                't_start' : v_segments['t-start'].tolist(),
                't_end': v_segments['t-end'].tolist(),
                'label': v_segments['label'].tolist(),
                'score': v_segments['score'].tolist()
            })
            mkdir_if_missing(os.path.join(ckpt_folder, 'unprocessed_results'))
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_preds_' + split_name), v_preds)
            np.save(os.path.join(ckpt_folder, 'unprocessed_results', 'v_gt_' + split_name), v_gt)
            v_results.to_csv(os.path.join(ckpt_folder, 'unprocessed_results', 'v_seg_' + split_name + '.csv'), index=False)

        # calculate validation metrics
        v_mAP, _ = det_eval.evaluate(v_segments)
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true')
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=0)
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=0)
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=0)
        
        # print results to terminal
        block1 = 'Epoch: [{:03d}/{:03d}]'.format(epoch, cfg['train_cfg']['epochs'])
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.mean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.mean(v_losses))
        block4 = ''
        block4  += '\t\tAvg. mAP {:>4.2f} (%) '.format(np.mean(np.nan_to_num(v_mAP) * 100))
        for tiou, tiou_mAP in zip(cfg['dataset']['tiou_thresholds'], np.nan_to_num(v_mAP)):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block4  += '\n\t\tAcc {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_acc) * 100))
        block4  += ' Prec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_prec) * 100))
        block4  += ' Rec {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_rec) * 100))
        block4  += ' F1 {:>4.2f} (%)'.format(np.mean(np.nan_to_num(v_f1) * 100))

        print('\n'.join([block1, block2, block3, block4]))

        if run is not None:
            run[split_name].append({"train_loss": np.mean(t_losses), "val_loss": np.mean(v_losses), "accuracy": np.mean(np.nan_to_num(v_acc)), "precision": np.mean(np.nan_to_num(v_prec)), "recall": np.mean(np.nan_to_num(v_rec)), 'f1': np.mean(np.nan_to_num(v_f1)), 'mAP': np.mean(np.nan_to_num(v_mAP))}, step=epoch)
            for tiou, tiou_mAP in zip(cfg['dataset']['tiou_thresholds'], np.nan_to_num(v_mAP)):
                run[split_name].append({'mAP@' + str(tiou): tiou_mAP}, step=epoch)    
    return t_losses, v_losses, v_mAP, v_preds, v_gt


def train_one_epoch(loader, network, opt, criterion, gpu=None):
    """
    Trains the network for one epoch using the given data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for loading the training data.
        network (torch.nn.Module): The network model to be trained.
        opt (torch.optim.Optimizer): The optimizer used for updating the network's parameters.
        criterion (torch.nn.Module): The loss function used for calculating the training loss.
        gpu (torch.device, optional): The GPU device to be used for training. Defaults to None.

    Returns:
        tuple: A tuple containing the trained network, a list of training losses, predicted labels, and ground truth labels.

    """
    losses, preds, gt = [], [], []
    network.train()
    for i, (ids, inputs, targets, weights, prob_targets) in enumerate(loader):
        if gpu is not None:
            inputs, targets, weights, prob_targets = inputs.to(gpu), targets.to(gpu), weights.to(gpu), prob_targets.to(gpu)
        output = network(inputs)
        _ = criterion(output, targets)
        batch_loss = torch.mean(criterion(output, targets) * weights.sum(axis=1))

        opt.zero_grad()
        batch_loss.backward()
        opt.step()
    
        # append train loss to list
        losses.append(batch_loss.item())

        # create predictions and append them to final list
        batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
        batch_gt = targets.cpu().numpy().flatten()
        preds = np.concatenate((preds, batch_preds))
        gt = np.concatenate((gt, batch_gt))
    
    return network, losses, preds, gt


def predict(loader, network, gpu=None):
    soft_preds = None
    network.train()
    for i, (ids, inputs, targets, weights, prob_targets) in enumerate(loader):
        
        if gpu is not None:
            inputs, targets, weights, prob_targets = inputs.to(gpu), targets.to(gpu), weights.to(gpu), prob_targets.to(gpu)
        softmax_output = nn.functional.softmax(network(inputs), dim=0)
    
        # create predictions and append them to final list
        batch_preds = softmax_output.cpu().detach().numpy()
        if soft_preds is None:
            soft_preds = batch_preds
        else:
            soft_preds = np.concatenate((soft_preds, batch_preds), axis=0)
    
    return soft_preds



def validate_one_epoch(loader, network, criterion, gpu=None):
    """
    Validate one epoch of the model.

    Args:
        loader (torch.utils.data.DataLoader): The validation data loader.
        network (torch.nn.Module): The network model.
        criterion (torch.nn.Module): The loss function.
        gpu (int, optional): The GPU device index. Defaults to None.

    Returns:
        tuple: A tuple containing the losses, predictions, and ground truth labels.

    """
    losses, preds, gt = [], [], []
    network.eval()
    with torch.no_grad():
        # iterate over validation dataset
        for _, (_, inputs, targets, _, _) in enumerate(loader):
            # send inputs through network to get predictions, loss and calculate softmax probabilities
            if gpu is not None:
                inputs, targets = inputs.to(gpu), targets.to(gpu)
            output = network(inputs)
            batch_loss = torch.mean(criterion(output, targets))
            losses.append(batch_loss.item())

            # create predictions and append them to final list
            batch_preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
            batch_gt = targets.cpu().numpy().flatten()
            preds = np.concatenate((preds, batch_preds))
            gt = np.concatenate((gt, batch_gt))
    return losses, preds, gt

