import torch
import torch.nn as nn
import scanpy as sc
import numpy as np

from tqdm import tqdm

from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from typing import List, Tuple
from .base import BaseTrainer
from ..dataset import CellTypeDataSet
from ..utils import divisible_by, cycle_dataloader, read_and_post_process_adata


@dataclass
class CellTypeClassifierTrainerConfig:
    # dataset name
    dataset_name: str

    # ann data
    ann_prc_data: str

    # output folder
    output_folder: str = "outputs"

    # model lr
    train_lr: float = 1e-4

    # morel adam
    adam_betas: Tuple[float, float] = (0.9, 0.99)

    # train step
    train_num_steps: int = 1000

    # model training batch size
    train_batch_size: int = 64

    # model evaluation batch size
    eval_batch_size: int = 64

    # eval and save model every
    save_and_eval_every: int = 1000

    # model cell type list
    cell_type_list: List[str] = field(
        default_factory=lambda: [
            "Ductal",
            "Ngn3 low EP",
            "Ngn3 high EP",
            "Pre-endocrine",
            "Beta",
            "Alpha",
            "Delta",
            "Epsilon",
        ]
    )

    # target cell type
    tgt_cell_type: str = "Beta"

    # training set ratio
    train_set_ratio: float = 0.8

    # hidden dim
    hidden_dim: int = 64

    # device
    num_workers: int = 0

    use_cuda: bool = True


class CellTypeClassifierTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        *,
        args: CellTypeClassifierTrainerConfig,
        has_wandb_writer: bool = False,
    ):
        super().__init__(args.output_folder, has_wandb_writer=has_wandb_writer)

        # device setting
        self.use_cuda = args.use_cuda

        # train
        self.dataset_name = args.dataset_name
        self.adata = read_and_post_process_adata(self.dataset_name, args.ann_prc_data)
        self.cell_type_dict = {k: v for v, k in enumerate(args.cell_type_list)}
        self.tgt_ind = self.cell_type_dict[args.tgt_cell_type]
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.save_and_eval_every = args.save_and_eval_every

        train_dataset, test_dataset = self.build_datasets(
            self.adata.X.A[:, self.adata.var["velocity_genes"]],
            self.adata.obs.clusters,
            train_set_ratio=args.train_set_ratio,
        )
        dataloader_worker = args.num_workers
        self.dl = cycle_dataloader(
            DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=dataloader_worker,
            )
        )
        self.eval_dl = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=dataloader_worker,
        )

        # model
        self.model = model.to(self.device)
        self.opt = Adam(
            self.model.parameters(), lr=args.train_lr, betas=args.adam_betas
        )

        # step counter state
        self.step = 0
        self.train_num_steps = args.train_num_steps

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt.load_state_dict(state["opt"])

    @property
    def device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    def build_datasets(self, data, label, train_set_ratio):
        # filter data not in given cell type list
        cell_type_list = list(self.cell_type_dict.keys())
        mask = np.isin(label, cell_type_list)
        data = data[mask]
        label = label[mask]

        # create dataset for head model
        cell_type_dataset = CellTypeDataSet(data, label, self.cell_type_dict)

        train_size = int(train_set_ratio * len(cell_type_dataset))
        test_size = len(cell_type_dataset) - train_size
        train_dataset, test_dataset = random_split(
            cell_type_dataset, [train_size, test_size]
        )
        return train_dataset, test_dataset

    @staticmethod
    @torch.inference_mode()
    def eval_acc(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.DeviceObjType,
        tgt_ind: int = None,
    ):
        pred_labels = []
        true_labels = []
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)

            pred_labels.append(pred.numpy(force=True))
            true_labels.append(labels.numpy(force=True))

        pred_arr = np.concatenate(pred_labels)
        true_arr = np.concatenate(true_labels)
        acc_over_all = np.mean(pred_arr == true_arr).item()

        acc_over_tgt = 0.0
        if tgt_ind is not None:
            acc_over_tgt = np.mean(
                pred_arr[true_arr == tgt_ind] == true_arr[true_arr == tgt_ind]
            ).item()

        results = {
            "acc_over_all": acc_over_all,
            "acc_over_tgt": acc_over_tgt,
        }
        return results

    @torch.inference_mode()
    def eval_during_training(self):
        self.model.eval()
        eval_results = self.eval_acc(
            self.model, self.eval_dl, self.device, tgt_ind=self.tgt_ind
        )
        self.model.train()
        return eval_results

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                gene, cell_type = next(self.dl)
                gene = gene.to(self.device)
                cell_type = cell_type.to(self.device)
                logit = self.model(gene)

                criterion = nn.CrossEntropyLoss()
                loss = criterion(logit, cell_type)
                loss.backward()

                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="cell_type_cls_train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="cell_type_cls_eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        log_results = self.eval_during_training()
        self.log(log_results, section="cell_type_cls_eval")
        self.save("final")
        print("cell type classifier training complete")
