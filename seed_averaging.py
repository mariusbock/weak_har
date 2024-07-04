# ------------------------------------------------------------------------
# Postprocessing script to calculate the average accuracy and F1 score across seeds
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------
import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# postprocessing parameters
dataset = 'actionsense'
network = 'deepconvlstm'
pred_type = 'weak_ce'
clusters = 100
if pred_type == 'baseline':
    path_to_preds = ['experiments/{}/{}/{}'.format(dataset, network, pred_type)]
else:
    path_to_preds = ['experiments/{}/{}/{}/{}'.format(dataset, network, pred_type, clusters)]
seeds = [1, 2, 3]

if dataset == 'wetlab':
    num_classes = 9
    sampling_rate = 50
    input_dim = 3
    json_files = [
    'data/wetlab/annotations/loso_sbj_0.json',
    'data/wetlab/annotations/loso_sbj_1.json',
    'data/wetlab/annotations/loso_sbj_2.json',
    'data/wetlab/annotations/loso_sbj_3.json',
    'data/wetlab/annotations/loso_sbj_4.json',
    'data/wetlab/annotations/loso_sbj_5.json',
    'data/wetlab/annotations/loso_sbj_6.json',
    'data/wetlab/annotations/loso_sbj_7.json',
    'data/wetlab/annotations/loso_sbj_8.json',
    'data/wetlab/annotations/loso_sbj_9.json',
    'data/wetlab/annotations/loso_sbj_10.json',
    'data/wetlab/annotations/loso_sbj_11.json',
    'data/wetlab/annotations/loso_sbj_12.json',
    'data/wetlab/annotations/loso_sbj_13.json',
    'data/wetlab/annotations/loso_sbj_14.json',
    'data/wetlab/annotations/loso_sbj_15.json',
    'data/wetlab/annotations/loso_sbj_16.json',
    'data/wetlab/annotations/loso_sbj_17.json',
    'data/wetlab/annotations/loso_sbj_18.json',
    'data/wetlab/annotations/loso_sbj_19.json',
    'data/wetlab/annotations/loso_sbj_20.json',
    'data/wetlab/annotations/loso_sbj_21.json'
    ]
elif dataset == 'wear':
    num_classes = 19
    sampling_rate = 50
    input_dim = 12
    json_files = [
    'data/wear/annotations/loso_sbj_0.json',
    'data/wear/annotations/loso_sbj_1.json',
    'data/wear/annotations/loso_sbj_2.json',
    'data/wear/annotations/loso_sbj_3.json',
    'data/wear/annotations/loso_sbj_4.json',
    'data/wear/annotations/loso_sbj_5.json',
    'data/wear/annotations/loso_sbj_6.json',
    'data/wear/annotations/loso_sbj_7.json',
    'data/wear/annotations/loso_sbj_8.json',
    'data/wear/annotations/loso_sbj_9.json',
    'data/wear/annotations/loso_sbj_10.json',
    'data/wear/annotations/loso_sbj_11.json',
    'data/wear/annotations/loso_sbj_12.json',
    'data/wear/annotations/loso_sbj_13.json',
    'data/wear/annotations/loso_sbj_14.json',
    'data/wear/annotations/loso_sbj_15.json',
    'data/wear/annotations/loso_sbj_16.json',
    'data/wear/annotations/loso_sbj_17.json'
    ]
elif dataset == 'actionsense':
    num_classes = 20
    sampling_rate = 50
    input_dim = 6
    json_files = [
    'data/actionsense/annotations/loso_sbj_0.json',
    'data/actionsense/annotations/loso_sbj_1.json',
    'data/actionsense/annotations/loso_sbj_2.json',
    'data/actionsense/annotations/loso_sbj_3.json',
    'data/actionsense/annotations/loso_sbj_4.json',
    'data/actionsense/annotations/loso_sbj_5.json',
    'data/actionsense/annotations/loso_sbj_6.json',
    'data/actionsense/annotations/loso_sbj_7.json',
    'data/actionsense/annotations/loso_sbj_8.json',
    ]
    
#print("Data Loading....")
for path in path_to_preds:
    all_acc = np.zeros((len(seeds), num_classes))
    all_f1 = np.zeros((len(seeds), num_classes))
    for s_pos, seed in enumerate(seeds):
        all_preds = np.array([])
        all_gt = np.array([])

        for i, j in enumerate(json_files):
            with open(j) as fi:
                file = json.load(fi)
                anno_file = file['database']
                if dataset == 'rwhar':
                    labels = list(file['label_dict'])
                else:
                    labels = ['null'] + list(file['label_dict'])
                label_dict = dict(zip(labels, list(range(len(labels)))))
                val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']
            
            v_data = np.empty((0, input_dim + 2))
                
            for sbj in val_sbjs:
                data = pd.read_csv(os.path.join('data/{}/raw/inertial'.format(dataset), sbj + '.csv'), index_col=False, low_memory=False).replace({"label": label_dict}).fillna(0).to_numpy()
                v_data = np.append(v_data, data, axis=0)
                
            v_preds = np.array([])
            v_orig_preds = np.load(os.path.join(path, 'seed_' + str(seed), 'unprocessed_results/v_preds_loso_sbj_{}.npy'.format(int(i))))
            
            for sbj in val_sbjs:
                sbj_pred = v_orig_preds[v_data[:, 0] == int(sbj.split("_")[-1])]
                v_preds = np.append(v_preds, sbj_pred)

            all_preds = np.concatenate((all_preds, v_preds))
            all_gt = np.concatenate((all_gt, v_data[:, -1]))
            #per class accuracy
            comb_conf = confusion_matrix(all_gt, all_preds, normalize='true', labels=range(0, num_classes))
            v_acc = comb_conf.diagonal()
            v_f1 = f1_score(v_data[:, -1], v_preds, average=None, labels=range(0, num_classes))

            all_acc[s_pos, :] += v_acc        
            all_f1[s_pos, :] += v_f1

            if seed == 1:
                comb_conf = np.around(comb_conf, 2)
                comb_conf[comb_conf == 0] = np.nan

                _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
                sns.heatmap(comb_conf, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens, cbar=False, annot_kws={
                            'fontsize': 16,
                        })
                ax.set_title('Confusion Matrix')
                pred_name = path.split('/')[-2]
                _.savefig(pred_name + ".pdf")
            #np.save("viz", all_preds)
                            
    print("Inidividual Accuracy:")
    for i in range(num_classes):
        print(np.around(np.mean(all_acc, axis=0)[i] / len(json_files), 4) * 100)

    print("Average Accuracy:")
    print("{:.4} (+/-{:.4})".format(np.mean(all_acc) / len(json_files) * 100, np.std(np.mean(all_acc, axis=1) / len(json_files)) * 100))

    print("Individual F1:")
    for i in range(num_classes):
        print(np.around(np.mean(all_f1, axis=0)[i] / len(json_files), 4) * 100)

    print("Average F1:")
    print("{:.4} (+/-{:.4})".format(np.mean(all_f1) / len(json_files) * 100, np.std(np.mean(all_f1, axis=1) / len(json_files)) * 100))