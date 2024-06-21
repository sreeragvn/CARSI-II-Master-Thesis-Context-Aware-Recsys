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
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchmetrics')

class Metric:
    """
    A class for computing and visualizing metrics and confusion matrices for model evaluation.
    """
    
    def __init__(self):
        """
        Initialize the Metric class by loading configurations and preparing metrics and confusion matrix.
        """
        self.metrics = configs['test']['metrics']
        self.k = configs['test']['k']
        
        with open(configs['train']['parameter_label_mapping_path'], 'rb') as f:
            self._label_mapping = pickle.load(f)
        
        self._num_classes = len(self._label_mapping)
        self._label_mapping['ignore'] = 0
        self._label_mapping = dict(sorted(self._label_mapping.items(), key=lambda item: item[1]))
        self.cm = MulticlassConfusionMatrix(num_classes=self._num_classes + 1, normalize='true').to(configs['device'])

    def metrics_calc_torch(self, target, output):
        """
        Calculate various metrics for the given targets and outputs using torchmetrics.
        
        Args:
            target (torch.Tensor): True labels.
            output (torch.Tensor): Model predictions.
        
        Returns:
            dict: Dictionary containing calculated metrics.
        """
        metrics = {metric: [] for metric in self.metrics}
        for k in self.k:
            for metric_name in self.metrics:
                if metric_name.lower() == 'f1score':
                    metric_func = torchmetrics.F1Score
                    metric = metric_func(num_classes=self._num_classes + 1, top_k=k, average='weighted', task='multiclass').to(configs['device'])
                elif metric_name.lower() == 'auroc':
                    metric_func = torchmetrics.AUROC
                    metric = metric_func(num_classes=self._num_classes + 1, average='weighted', task='multiclass').to(configs['device'])
                else:
                    metric_func = getattr(torchmetrics, metric_name.capitalize())
                    metric = metric_func(num_classes=self._num_classes + 1, top_k=k, average='weighted', task='multiclass').to(configs['device'])
                value = metric(output, target)
                metrics[metric_name].append(round(value.item(), 2))
        return metrics

    def eval_new(self, model, test_dataloader, test=False):
        """
        Evaluate the model on the test dataset and calculate metrics and confusion matrix.
        
        Args:
            model (torch.nn.Module): The model to evaluate.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            test (bool): Flag to indicate if it is a test evaluation.
        
        Returns:
            dict: Dictionary containing calculated metrics.
            PIL.Image: Confusion matrix image.
        """
        true_labels_list = []
        pred_scores_list = []

        for _, tem in enumerate(test_dataloader):
            if not isinstance(tem, list):
                tem = [tem]
            batch_data = list(map(lambda x: x.long().to(configs['device']) if not isinstance(x, list) 
                                  else torch.stack([t.float().to(configs['device']) for t in x], dim=1), tem))
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
        computed_confusion = np.nan_to_num(computed_confusion, nan=0.0)
        im = self.plot_confusion_matrix(computed_confusion)
        
        return metrics_data, im

    def eval(self, model, test_dataloader, test=False):
        """
        Evaluate the model and return metrics and confusion matrix.
        
        Args:
            model (torch.nn.Module): The model to evaluate.
            test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            test (bool): Flag to indicate if it is a test evaluation.
        
        Returns:
            dict: Dictionary containing calculated metrics.
            PIL.Image: Confusion matrix image.
        """
        metrics_data, cm_im = self.eval_new(model, test_dataloader, test)
        return metrics_data, cm_im
    
    def plot_confusion_matrix(self, computed_confusion):
        """
        Plot confusion matrix.
        
        Args:
            computed_confusion (np.ndarray): Computed confusion matrix.
        
        Returns:
            torch.Tensor: Confusion matrix image as a tensor.
        """
        df_cm = pd.DataFrame(computed_confusion, index=self._label_mapping.values(), columns=self._label_mapping.values())
        df_cm = df_cm.iloc[1:-1, 1:-1]  # Exclude 'ignore' class from the plot
        
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1)

        sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='.2f', ax=ax, cmap='Greens', cbar=False)
        
        for i in range(len(df_cm)):
            for j in range(len(df_cm)):
                value = df_cm.iloc[i, j]
                rect = Rectangle((j, i), 1, 1, fill=False, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                if value != 0 and i != j:
                    alpha = min(0.35 + 0.7 * (value / df_cm.values.max()), 1.0)
                    ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color='#9ecae1', alpha=alpha))
                elif i == j:
                    alpha = min(0.35 + 0.7 * (value / df_cm.values.max()), 1.0)
                    ax.add_patch(Rectangle((i, j), 1, 1, fill=True, color='#2ca25f', alpha=alpha))

        for text in ax.texts:
            text.set_color('black')
            text.set_ha('center')
            text.set_va('center')

        num_labels = len(self._label_mapping) - 2
        rect = Rectangle((0, 0), num_labels, num_labels, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        excluded_keys = {0, 23}
        _label_mapping_display = {key: value for key, value in self._label_mapping.items() if value not in excluded_keys}

        ax.legend(_label_mapping_display.values(), _label_mapping_display.keys(), handler_map={int: self.IntHandler()}, loc='upper left', bbox_to_anchor=(1.01, 1))
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        im = Image.open(buf)
        im = transforms.ToTensor()(im)
        
        return im

    class IntHandler:
        """
        A handler for legend entries that are integers.
        """
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            x0, y0 = handlebox.xdescent, handlebox.ydescent
            text = plt.matplotlib.text.Text(x0, y0, str(orig_handle))
            handlebox.add_artist(text)
            return text
