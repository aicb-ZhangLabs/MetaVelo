from math import ceil
import torch
import torch.nn as nn
import wandb
import scanpy as sc
import numpy as np

from tqdm import tqdm
from typing import Callable, List, Tuple
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from einops import rearrange
from dataclasses import dataclass, asdict, field

from ..pert.sampling import SubsetSamplingPertModule
from ..eval import eval_cell_type_ratio, eval_cell_type
from ..utils import (
    divisible_by,
    sample_vector_field,
    visualize_pert_categorical_distribution,
    visualize_trajs,
    visualize_vector_field,
    cycle_dataloader,
    read_and_post_process_adata,
)
from ..trajectory.sample import sample_trajectory, sample_deep_velo_trajectory_from_ckpt
from ..head.base import HeadModule
from ..dataset import GeneTrajDataset
from ..pert import PertModule
from ..train.base import BaseTrainer


@dataclass
class TrainerConfig:
    # dataset name
    dataset_name: str

    # train dataset path
    data_folder: str

    # output path
    output_folder: str

    # ann data
    ann_prc_data: str

    # ann raw data
    ann_raw_data: str

    # deep velo model checkpoint
    deep_velo_model_checkpoint: str

    # deep velo model dim
    deep_velo_model_dim: int

    # eval dataset path
    eval_data_folder: str = ""

    # test dataset path
    test_data_folder: str = ""

    # deep velo scaling factor
    deep_velo_scaling_factor: float = 0.8170944342283816

    # deep velo intermediate step for integration
    deep_velo_intermediate_step: int = 3

    # optimization
    train_batch_size: int = 16

    train_lr: float = 1e-4

    train_num_steps: int = 100000

    save_and_sample_every: int = 1000

    max_grad_norm: float = 1

    adam_betas: Tuple[float, float] = (0.9, 0.99)

    # time span
    t_span: Tuple[int, int] = (0, 1)

    # total step for integration (t_step - 1 must be divisible by seq_step - 1)
    t_step: int = 25

    # sequence step
    seq_step: int = 5

    # starting cell type
    starting_cell_type: str = "Pre-endocrine"

    # only use the starting cell type data
    train_only_starting_cell_type_data: bool = True

    # do balance processing on dataset
    balance_dataset: bool = False

    # sample ratio for balance processing
    sample_ratio: float = 0.65

    # device
    num_workers: int = 0

    use_cuda: bool = True

    # evaluation
    num_samples: int = 25

    # evaluation cell type ratio
    cell_type_ratio_keys: List[str] = field(default_factory=lambda: ["Alpha", "Beta"])

    # random seed
    seed: int = 2023


class Trainer(BaseTrainer):
    def __init__(
        self, model: nn.Module, *, args: TrainerConfig, has_wandb_writer: bool = False
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer=has_wandb_writer)

        self.model = model
        self.args = args

        self.batch_size = args.train_batch_size
        self.dataset_name = args.dataset_name
        self.adata = read_and_post_process_adata(self.dataset_name, args.ann_prc_data)
        self.adata_raw = sc.read_h5ad(args.ann_raw_data)

        if self.args.balance_dataset:
            raw_tra = np.load(args.data_folder)["path"]
            final_cell_states = eval_cell_type(
                self.adata.X.A[:, self.adata.var["velocity_genes"]],
                self.adata.obs.clusters,
                raw_tra[self.args.seq_step - 1],
                random_state=self.args.seed,
            )
        else:
            final_cell_states = None

        self.ds = GeneTrajDataset(
            args.data_folder,
            starting_cell_type=(
                self.args.starting_cell_type
                if args.train_only_starting_cell_type_data
                else None
            ),
            global_cell_type_array=self.adata.obs.clusters.to_numpy(),
            balance=self.args.balance_dataset,
            final_cell_types=final_cell_states,
            sample_ratio=self.args.sample_ratio,
        )
        dl = DataLoader(
            self.ds,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        self.dl = dl

        self.eval_ds = GeneTrajDataset(
            args.eval_data_folder if args.eval_data_folder else args.data_folder,
            starting_cell_type=(
                self.args.starting_cell_type
                if args.train_only_starting_cell_type_data
                else None
            ),
            global_cell_type_array=self.adata.obs.clusters.to_numpy(),
        )
        self.eval_dl = DataLoader(
            self.eval_ds, batch_size=self.batch_size, shuffle=False
        )

        self.test_ds = GeneTrajDataset(
            args.test_data_folder if args.test_data_folder else args.data_folder,
            starting_cell_type=(
                self.args.starting_cell_type
                if args.train_only_starting_cell_type_data
                else None
            ),
            global_cell_type_array=self.adata.obs.clusters.to_numpy(),
        )
        self.test_dl = DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )

        # step counter state
        self.step = 0
        self.train_num_steps = args.train_num_steps
        self.save_and_sample_every = args.save_and_sample_every

        self.model = model.to(self.device)
        self.t_span = torch.linspace(*args.t_span, args.t_step, device=self.device)
        self.opt = Adam(model.parameters(), lr=args.train_lr, betas=args.adam_betas)
        # self.opt = SGD(model.parameters(), lr=args.train_lr)

        assert (args.t_step - 1) % (
            args.seq_step - 1
        ) == 0, "t_step is incompatible with seq_step"
        self.t_offset = (args.t_step - 1) // (args.seq_step - 1)

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "args": asdict(self.args),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt.load_state_dict(state["opt"])

    @property
    def device(self):
        return torch.device("cuda") if self.args.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    @torch.inference_mode()
    def generate_trajectory(self, dataloader: DataLoader):
        grd_trajs, all_trajs = sample_trajectory(
            self.model,
            self.t_span,
            self.args.seq_step,
            self.t_offset,
            dataloader,
            self.device,
        )
        return grd_trajs, all_trajs

    @torch.inference_mode()
    def generate_reference_trajectory(
        self,
        dataloader: DataLoader,
    ) -> tuple[None, torch.Tensor]:
        list_of_initial_state = []
        for data in dataloader:
            data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                self.device
            )
            inp = data[0]
            list_of_initial_state.append(inp)
        initial_state = torch.cat(list_of_initial_state, dim=0)

        all_trajs, _ = sample_deep_velo_trajectory_from_ckpt(
            self.adata,
            self.adata_raw,
            self.args.deep_velo_model_checkpoint,
            model_input_dim=self.args.deep_velo_model_dim,
            initial_state=initial_state,
            scaling_factor=self.args.deep_velo_scaling_factor,
            intermediate_step=self.args.deep_velo_intermediate_step,
            device=self.device,
            max_steps=self.args.seq_step - 1,
            pre_log1p=True,
            random_state=self.args.seed,
        )

        all_trajs = np.log1p(all_trajs.astype(np.float32))
        return None, torch.from_numpy(all_trajs).to(self.device)

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        # turn model to eval mode
        self.set_model_state(train=False)

        # eval cell type ratio
        raw_genes = self.adata.X.A[:, self.adata.var["velocity_genes"]]
        cell_types = self.adata.obs.clusters

        # eval trajectory from ode and reference
        grd_trajs, all_trajs = self.generate_trajectory(dataloader)
        _, ref_all_trajs = self.generate_reference_trajectory(dataloader)
        grd_trajs, all_trajs = grd_trajs.numpy(force=True), all_trajs.numpy(force=True)
        ref_all_trajs = ref_all_trajs.numpy(force=True)

        # eval cell type ratio
        log_results = {}
        for eval_type, trajs in {
            "ratio": all_trajs,
            "deep_velo_ratio": ref_all_trajs,
        }.items():
            ratio_results = eval_cell_type_ratio(
                raw_genes,
                cell_types,
                trajs[-1],
                self.args.cell_type_ratio_keys,
                random_state=self.args.seed,
            )
            log_results.update(
                {
                    f"{eval_type}/cell_type_{k}_ratio": v
                    for k, v in ratio_results.items()
                }
            )

        # visualize trajs
        cell_colors = self.adata.uns["clusters_colors"][
            self.adata.obs.clusters.cat.codes.to_numpy()
        ]
        grd_fig, _ = visualize_trajs(
            raw_genes,
            cell_colors,
            grd_trajs[:, : self.args.num_samples],
        )
        all_fig, _ = visualize_trajs(
            raw_genes,
            cell_colors,
            all_trajs[:, : self.args.num_samples],
        )
        log_results.update(
            {
                "trajectory/sampled_trajectory": wandb.Image(all_fig),
                "trajectory/ground_trajectory": wandb.Image(grd_fig),
            }
        )

        # evaluate velocity field
        for i in range(self.args.seq_step):
            offset = self.t_offset * i
            vec = sample_vector_field(
                self.model.vf,
                t=offset,
                query_positions=raw_genes,
                device=self.device,
            )
            fig, _ = visualize_vector_field(self.adata, velocity=vec.cpu().numpy())
            log_results.update(
                {
                    f"vector_field/neural_field_at_step{i}": wandb.Image(fig),
                }
            )

        fig, _ = visualize_vector_field(
            self.adata,
            velocity=self.adata.layers["velocity"][:, self.adata.var["velocity_genes"]],
        )
        log_results.update(
            {
                "vector_field/scvelo_field": wandb.Image(fig),
            }
        )

        return log_results

    def set_model_state(self, train: bool = True):
        """In this trainer, we only train on the model.

        :param train: whether to train mode, defaults to True
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

    @torch.inference_mode()
    def eval_during_training(self):
        results = self.eval(self.eval_dl)
        self.set_model_state(train=True)
        return results

    def train(self):
        self.set_model_state(train=True)

        device = self.device
        loss_fn = nn.MSELoss()
        dl = cycle_dataloader(self.dl)
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
        ) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                data = next(dl)
                data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                    device
                )
                inp = data[0]

                _, pred_traj = self.model(inp, self.t_span)
                loss = loss_fn(pred_traj[:: self.t_offset][1:], data[1:])  # shift right
                total_loss += loss.item()
                loss.backward()

                pbar.set_description(f"loss: {total_loss:.4f}")
                self.log({"loss": total_loss}, section="train")

                # accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.args.max_grad_norm
                )

                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                if self.step != 0 and divisible_by(
                    self.step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="eval")
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")
        print("training complete")


@dataclass
class PerturbationTrainerConfig(TrainerConfig):
    # perturbation total step
    pert_num_steps: int = 100000

    # perturbation state step
    pert_state_step: int = -1


class PerturbationTrainer(Trainer):
    def __init__(
        self,
        pert_model: PertModule,
        model: nn.Module,
        head_model: HeadModule,
        *,
        args: PerturbationTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(model, args=args, has_wandb_writer=has_wandb_writer)
        self.pert_args = args

        self.pert_model = pert_model.to(self.device)
        self.head_model = head_model.to(self.device)

        # dummy optimizer for clean gradient on head model
        self.head_model_opt = SGD(self.head_model.parameters(), lr=1e-3)

        # step counter state
        self.pert_step = 0
        self.pert_num_steps = self.pert_args.pert_num_steps

    def get_state(self):
        parent_state = super().get_state()
        state = {
            "pert_step": self.pert_step,
            "pert_model": self.pert_model.state_dict(),
            "pert_args": asdict(self.pert_args),
            "head_model": self.head_model.state_dict(),
            "parent_state": parent_state,
        }
        return state

    def load_state(self, state):
        super().load_state(state["parent_state"])
        self.pert_step = state["pert_step"]
        self.pert_model.load_state_dict(state["pert_model"])
        self.head_model.load_state_dict(state["head_model"])

    @property
    def global_step(self) -> int:
        return self.pert_step

    @torch.inference_mode()
    def generate_trajectory(self, dataloader: DataLoader):
        grd_trajs, all_trajs = sample_trajectory(
            self.model,
            self.t_span,
            self.args.seq_step,
            self.t_offset,
            dataloader,
            self.device,
            pert_model=self.pert_model,
        )
        return grd_trajs, all_trajs

    @torch.inference_mode()
    def generate_reference_trajectory(
        self,
        dataloader: DataLoader,
    ) -> tuple[None, torch.Tensor]:
        list_of_initial_state = []
        for data in dataloader:
            data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                self.device
            )
            inp = data[0]

            # add perturbation
            inp = self.pert_model.perturb(inp)

            list_of_initial_state.append(inp)
        initial_state = torch.cat(list_of_initial_state, dim=0)

        all_trajs, _ = sample_deep_velo_trajectory_from_ckpt(
            self.adata,
            self.adata_raw,
            self.args.deep_velo_model_checkpoint,
            model_input_dim=self.args.deep_velo_model_dim,
            initial_state=initial_state,
            scaling_factor=self.args.deep_velo_scaling_factor,
            intermediate_step=self.args.deep_velo_intermediate_step,
            device=self.device,
            max_steps=self.args.seq_step - 1,
            pre_log1p=True,
            random_state=self.args.seed,
        )

        all_trajs = np.log1p(all_trajs.astype(np.float32))
        return None, torch.from_numpy(all_trajs).to(self.device)

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        # turn model to eval mode
        self.set_model_state(train=False)

        # eval cell type ratio
        raw_genes = self.adata.X.A[:, self.adata.var["velocity_genes"]]
        cell_types = self.adata.obs.clusters

        # eval trajectory from ode and reference
        grd_trajs, all_trajs = super().generate_trajectory(dataloader)
        _, ref_all_trajs = super().generate_reference_trajectory(dataloader)
        grd_trajs, all_trajs = grd_trajs.numpy(force=True), all_trajs.numpy(force=True)
        ref_all_trajs = ref_all_trajs.numpy(force=True)

        log_results = {}
        for eval_type, trajs in {
            "ratio": all_trajs,
            "deep_velo_ratio": ref_all_trajs,
        }.items():
            ratio_results = eval_cell_type_ratio(
                raw_genes,
                cell_types,
                trajs[-1],
                self.args.cell_type_ratio_keys,
                random_state=self.pert_args.seed,
            )
            log_results.update(
                {
                    f"{eval_type}/cell_type_{k}_ratio": v
                    for k, v in ratio_results.items()
                }
            )

        # eval trajectory from ode and reference after perturbation
        _, pert_all_trajs = self.generate_trajectory(dataloader)
        _, pert_ref_all_trajs = self.generate_reference_trajectory(dataloader)
        pert_all_trajs, pert_ref_all_trajs = pert_all_trajs.numpy(
            force=True
        ), pert_ref_all_trajs.numpy(force=True)

        # eval cell type ratio
        for eval_type, trajs in {
            "ratio": pert_all_trajs,
            "deep_velo_ratio": pert_ref_all_trajs,
        }.items():
            pert_ratio_results = eval_cell_type_ratio(
                raw_genes,
                cell_types,
                trajs[-1],
                self.args.cell_type_ratio_keys,
                random_state=self.pert_args.seed,
            )
            log_results.update(
                {
                    f"{eval_type}/cell_type_{k}_ratio_pert": v
                    for k, v in pert_ratio_results.items()
                }
            )

        # visualize perturbations for sampling perturbation
        if isinstance(self.pert_model, SubsetSamplingPertModule):
            weight: torch.Tensor = torch.softmax(
                self.pert_model.sampler_model.get_weights(), dim=-1
            )
            pert_vec, khot = self.pert_model.sampler_model.get_pert_vec()
            distribution_fig, _ = visualize_pert_categorical_distribution(
                weight.numpy(force=True),
                khot.numpy(force=True),
                pert_vec.numpy(force=True),
            )
            log_results.update(
                {"distribution/pert_distribution": wandb.Image(distribution_fig)}
            )

        # visualize trajs
        cell_colors = self.adata.uns["clusters_colors"][
            self.adata.obs.clusters.cat.codes.to_numpy()
        ]
        grd_fig, _ = visualize_trajs(
            raw_genes,
            cell_colors,
            grd_trajs[:, : self.args.num_samples],
        )
        all_fig, _ = visualize_trajs(
            raw_genes,
            cell_colors,
            all_trajs[:, : self.args.num_samples],
        )
        pert_all_fig, _ = visualize_trajs(
            raw_genes,
            cell_colors,
            pert_all_trajs[:, : self.args.num_samples],
        )
        log_results.update(
            {
                "trajectory/ground_trajectory": wandb.Image(grd_fig),
                "trajectory/sampled_trajectory": wandb.Image(all_fig),
                "trajectory/sampled_trajectory_pert": wandb.Image(pert_all_fig),
            }
        )

        return log_results

    def get_forward_fn(self):
        def forward_fn(inp: torch.Tensor) -> torch.Tensor:
            _, pred_traj = self.model(inp, self.t_span)

            final_state = pred_traj[:: self.t_offset][self.pert_args.pert_state_step]
            return final_state

        return forward_fn

    def get_per_sample_eval_fn(self):
        @torch.inference_mode()
        def eval_fn(final_state: torch.Tensor) -> torch.Tensor:
            loss = self.head_model.get_loss(final_state, reduction="none")
            return loss

        return eval_fn

    def set_model_state(self, train: bool = True):
        """In this trainer, we only train on the perturbation model.

        :param train: whether to train mode, defaults to True
        """
        self.model.eval()
        self.head_model.eval()
        if train:
            self.pert_model.train()
        else:
            self.pert_model.eval()

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        dl = cycle_dataloader(self.dl)
        with tqdm(
            initial=self.pert_step,
            total=self.pert_num_steps,
        ) as pbar:
            while self.pert_step < self.pert_num_steps:
                data = next(dl)
                data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                    self.device
                )
                inp = data[0]

                pert_inp = self.pert_model.perturb(
                    inp,
                    step=self.pert_step,
                    update_tau=True,
                )
                final_state = self.get_forward_fn()(pert_inp)
                loss = self.head_model.get_loss(final_state)
                loss.backward()

                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="pert")

                # update perturbation optimizaer, and clean up model's gradient
                self.pert_model.update(
                    inp,
                    forward_fn=self.get_forward_fn(),
                    per_sample_eval_fn=self.get_per_sample_eval_fn(),
                )
                self.opt.zero_grad()
                self.head_model.zero_grad()

                self.pert_step += 1
                if self.pert_step != 0 and divisible_by(
                    self.pert_step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="eval")
                    milestone = self.pert_step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")
        print("perturbation complete")


class NonIterativePerturbationTrainer(PerturbationTrainer):
    """A helper class for non-iterative perturbation trainer, which contains the
    according tensor dataset."""

    def get_tensor_dataset(self, dl: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_model_state(train=False)

        xs = []
        ys = []
        for data in dl:
            data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                self.device
            )
            inp = data[0]
            final_state = self.get_forward_fn()(inp)
            xs.append(inp)
            ys.append(final_state)

        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        return x, y

    @torch.inference_mode()
    def evaluate_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.set_model_state(train=False)
        final_state = self.get_forward_fn()(x)
        return final_state

    def evaluate_per_sample_head_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        self.set_model_state(train=False)
        loss = self.get_per_sample_eval_fn()(x)
        return loss

    def evaluate_forward_and_per_head_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.set_model_state(train=False)
        final_state = self.evaluate_forward(x)
        loss = self.evaluate_per_sample_head_loss(final_state, y)
        return loss


class SAGEPerturbationTrainer(NonIterativePerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x, y = self.get_tensor_dataset(self.dl)
        self.pert_model.update(
            x=x,
            y=y,
            forward_fn=self.evaluate_forward,
            device=self.device,
            head_loss_fn=self.head_model.get_loss,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class CXPlainPerturbationTrainer(NonIterativePerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x, y = self.get_tensor_dataset(self.dl)
        x_eval, y_eval = self.get_tensor_dataset(self.eval_dl)

        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            x_eval=x_eval,
            forward_fn=self.evaluate_forward,
            device=self.device,
            head_loss_fn=self.evaluate_per_sample_head_loss,
            mask_value=self.pert_model.get_pert_val().detach().clone().unsqueeze(0),
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class LIMEPerturbationTrainer(NonIterativePerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x, y = self.get_tensor_dataset(self.dl)

        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            forward_fn=self.evaluate_forward_and_per_head_loss,
            mask_value=self.pert_model.get_pert_val().detach().clone(),
            label=0,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class FIMAPPerturbationTrainer(NonIterativePerturbationTrainer):
    def get_pert_ind(self):
        x, _ = self.get_tensor_dataset(self.dl)
        return self.pert_model.get_pert_ind(x=x)

    @torch.inference_mode()
    def generate_trajectory(self, dataloader: DataLoader):
        global_pert_ind = self.get_pert_ind()
        grd_trajs, all_trajs = sample_trajectory(
            self.model,
            self.t_span,
            self.args.seq_step,
            self.t_offset,
            dataloader,
            self.device,
            pert_model=self.pert_model,
            pert_model_kwargs={"pert_ind": global_pert_ind},
        )
        return grd_trajs, all_trajs

    @torch.inference_mode()
    def generate_reference_trajectory(
        self,
        dataloader: DataLoader,
    ) -> tuple[None, torch.Tensor]:
        global_pert_ind = self.get_pert_ind()
        list_of_initial_state = []
        for data in dataloader:
            data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                self.device
            )
            inp = data[0]

            # add perturbation
            inp = self.pert_model.perturb(inp, pert_ind=global_pert_ind)

            list_of_initial_state.append(inp)
        initial_state = torch.cat(list_of_initial_state, dim=0)

        all_trajs, _ = sample_deep_velo_trajectory_from_ckpt(
            self.adata,
            self.adata_raw,
            self.args.deep_velo_model_checkpoint,
            model_input_dim=self.args.deep_velo_model_dim,
            initial_state=initial_state,
            scaling_factor=self.args.deep_velo_scaling_factor,
            intermediate_step=self.args.deep_velo_intermediate_step,
            device=self.device,
            max_steps=self.args.seq_step - 1,
            pre_log1p=True,
            random_state=self.args.seed,
        )

        all_trajs = np.log1p(all_trajs.astype(np.float32))
        return None, torch.from_numpy(all_trajs).to(self.device)


class PermutationPerturbationTrainer(NonIterativePerturbationTrainer):
    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        x, y = self.get_tensor_dataset(self.dl)

        self.set_model_state(train=False)
        self.pert_model.update(
            x=x,
            forward_fn=self.evaluate_forward_and_per_head_loss,
        )

        self.pert_step += 1
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")

        self.save("final")
        print("perturbation complete")


class SaliencyPerturbationTrainer(PerturbationTrainer):
    def _differntiable_loss_eval(self, inp: torch.Tensor):
        final_state = self.get_forward_fn()(inp)
        loss = self.head_model.get_loss(final_state)
        return loss

    def train(self):
        self.set_model_state(train=True)

        # eval before perturbation
        log_results = self.eval_during_training()
        self.log(log_results, section="eval")

        # start perturbating
        dl = self.dl
        with tqdm(
            initial=self.pert_step,
            total=ceil(len(dl) // self.batch_size),
        ) as pbar:
            for data in dl:
                data = rearrange(data, "b t h -> t b h")[: self.args.seq_step].to(
                    self.device
                )
                inp = data[0]

                # update perturbation optimizaer, and clean up model's gradient
                self.pert_model.update(
                    inp,
                    forward_fn=self._differntiable_loss_eval,
                )
                self.opt.zero_grad()
                self.head_model.zero_grad()

                self.pert_step += 1
                if self.pert_step != 0 and divisible_by(
                    self.pert_step, self.save_and_sample_every
                ):
                    log_results = self.eval_during_training()
                    self.log(log_results, section="eval")
                    milestone = self.pert_step // self.save_and_sample_every
                    self.save(milestone)

                pbar.update(1)

        self.save("final")
        log_results = self.eval(self.test_dl)
        self.log(log_results, section="test")
        print("perturbation complete")
