import io
import itertools
from matplotlib.patches import Rectangle
import torch
import numpy as np
from config.configurator import configs
import pandas as pd
import os
import pickle
from torchmetrics.classification import MulticlassConfusionMatrix
import torchmetrics
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import seaborn as sn
import matplotlib.colors as mcolors

class Metric(object):
    def __init__(self):
        self.metrics = configs['test']['metrics']
        self.k = configs['test']['k']
        with open(configs['train']['parameter_label_mapping_path'], 'rb') as f:
            self._label_mapping = pickle.load(f)
        self._num_classes = len(self._label_mapping)
        # self.class_mapping = self._label_mapping
        self._label_mapping['ignore'] = 0
        self._label_mapping = dict(sorted(self._label_mapping.items(), key=lambda item: item[1]))
        self.cm = MulticlassConfusionMatrix(num_classes=self._num_classes + 1).to(configs['device'])

    def metrics_calc_torch(self, target, output):
        metrics = {metric: [] for metric in self.metrics}
        for k in self.k:
            for metric_name in self.metrics:
                if metric_name.lower() == 'f1score':
                    metric_func = torchmetrics.F1Score
                else:
                    metric_func = getattr(torchmetrics, metric_name.capitalize())
                metric = metric_func(num_classes=self._num_classes + 1, top_k=k, average='weighted', task='multiclass').to(configs['device'])
                value = metric(output, target)
                metrics[metric_name].append(round(value.item(), 2))
        return metrics

    def eval_new(self, model, test_dataloader, test=False):
        true_labels_list = []
        pred_scores_list = []

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            batch_data = list(map(lambda x: x.long().to(configs['device']) if not isinstance(x, list) 
                                  else torch.stack([t.float().to(configs['device']) for t in x], dim=1)
                                  , tem))
            _, _, batch_last_items, _, _, _, _, _  = batch_data
            with torch.no_grad():
                batch_pred = model.full_predict(batch_data)
            true_labels_list.append(batch_last_items)
            pred_scores_list.append(batch_pred)

        true_labels = torch.cat(true_labels_list, dim=0)
        pred_scores = torch.cat(pred_scores_list, dim=0)

        metrics_data = self.metrics_calc_torch(true_labels, pred_scores)
        self.cm(pred_scores, true_labels)
        computed_confusion = self.cm(pred_scores, true_labels).cpu().numpy()
        im = self.plot_confusion_matrix(computed_confusion)
            # self.writer.add_image(f"confusion_matrix/{configs['test']['data']}", im)
            # cm_name = configs['test']['save_path']
            # file_path = 'results_metrics/'
            # if not os.path.exists(file_path):
            #     os.makedirs(file_path)
            # np.savetxt(file_path+f'cm_{cm_name}.csv', computed_confusion, delimiter=',', fmt='%d')
        return metrics_data, im

    def eval(self, model, test_dataloader, test=False):
        metrics_data, cm_im = self.eval_new(model, test_dataloader, test)
        return metrics_data, cm_im

    # def plot_confusion_matrix(self, computed_confusion):
    #     """
    #     Plot confusion matrix.
    #     """
    #     df_cm = pd.DataFrame(
    #             computed_confusion,
    #             index=self._label_mapping.values(),
    #             columns=self._label_mapping.values(),
    #         )
    #     fig, ax = plt.subplots(figsize=(15, 7))
    #     fig.subplots_adjust(left=0.05, right=.65)
    #     sn.set(font_scale=1.2)

    #     # Plot the confusion matrix without a heatmap palette
    #     sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', ax=ax, cmap='Greens', cbar=False)

    #     # Loop through the diagonal elements to add green background
    #     for i in range(len(df_cm)):
    #         ax.add_patch(Rectangle((i, i), 1, 1, fill=True, color='green', alpha=0.3))

    #     # Loop through the texts to set text color
    #     for text in ax.texts:
    #         text.set_color('black')  # Set text color to black

    #     ax.legend(
    #         self._label_mapping.values(),
    #         self._label_mapping.keys(),
    #         handler_map={int: self.IntHandler()},
    #         loc='upper left',
    #         bbox_to_anchor=(1.2, 1)
    #     )
    #     buf = io.BytesIO()
    #     plt.savefig(buf, format='jpeg', bbox_inches='tight')
    #     plt.close() 
    #     buf.seek(0)
    #     im = Image.open(buf)
    #     im = transforms.ToTensor()(im)
    #     return im

    def plot_confusion_matrix(self, computed_confusion):
        """
        Plot confusion matrix.
        """
        df_cm = pd.DataFrame(
            computed_confusion,
            index=self._label_mapping.values(),
            columns=self._label_mapping.values(),
        )
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)

        # Plot the confusion matrix without a heatmap palette
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='d', ax=ax, cmap='Greens', cbar=False)

        # Loop through the elements to add green background for non-zero numbers
        for i in range(len(df_cm)):
            ax.add_patch(Rectangle((i, i), 1, 1, fill=True, color='green', alpha=0.3))
            for j in range(len(df_cm)):
                value = df_cm.iloc[i, j]
                if value != 0 and i !=j:
                    # Calculate darkness of green based on the numerical value
                    alpha = min(0.3 + 0.7 * (value / df_cm.values.max()), 1.0)
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color=(1, 1, 0, alpha)))
                elif i ==j:
                    ax.add_patch(Rectangle((i, j), 1, 1, fill=True, color=(0, 1, 0), alpha=0.5))


        # Loop through the texts to set text color
        for text in ax.texts:
            text.set_color('black')  # Set text color to black

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        ax.legend(
            self._label_mapping.values(),
            self._label_mapping.keys(),
            handler_map={int: self.IntHandler()},
            loc='upper left',
            bbox_to_anchor=(1.01, 1)
        )
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        plt.close() 
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        return im

    class IntHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
            handlebox.add_artist(text)
            return text
