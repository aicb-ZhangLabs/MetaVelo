from src.train.synthetic_data import (
    SyntheticDataClassifierTrainerConfig,
)
from src.trainer.synthetic_data import SyntheticDataPerturbationTrainerConfig
from src.slurm import SlurmConfig
from src.pert import (
    PertModuleConfig,
    SubsetSamplingPertModuleConfig,
    SAGEPertModuleConfig,
    LIMEPertModuleConfig,
    FIMAPPertModuleConfig,
    CXPlainPertModuleConfig,
    PermutationPertModuleConfig,
    SaliencyPertModuleConfig,
    ApproximationPertModuleConfig,
)
from src.pert.cxplain.train import CXPlainTrainerConfig
from configs.utils import get_output_path

output_path: str = get_output_path()

nonlinear_additive_head_trainer = SyntheticDataClassifierTrainerConfig(
    data_folder="datasets/synthetic_data",
    output_folder=f"{output_path}/head_model",
    synthetic_dataset_name="nonlinear_additive",
    overwrite_synthetic_dataset=False,
    num_type_list=["0", "1"],
    tgt_num_type="0",
    synthetic_data_size=10000,
    synthetic_data_feature_dim=1000,
    train_num_steps=5000,
)

orange_skin_additive_head_trainer = SyntheticDataClassifierTrainerConfig(
    data_folder="datasets/synthetic_data",
    output_folder=f"{output_path}/head_model",
    synthetic_dataset_name="orange_skin_additive",
    overwrite_synthetic_dataset=True,
    num_type_list=["0", "1"],
    tgt_num_type="0",
    synthetic_data_size=10000,
    synthetic_data_feature_dim=1000,
    train_num_steps=5000,
)

nonlinear_additive_perturb_trainer = SyntheticDataPerturbationTrainerConfig(
    data_folder="datasets/synthetic_data",
    output_folder=output_path,
    synthetic_dataset_name="nonlinear_additive",
    starting_num_type="1",
    synthetic_data_size=1000,
    synthetic_data_feature_dim=10,
    pert_num_steps=500,
    save_and_sample_every=100,
    num_workers=4,
)

orange_skin_additive_perturb_trainer = SyntheticDataPerturbationTrainerConfig(
    data_folder="datasets/synthetic_data",
    output_folder=output_path,
    synthetic_dataset_name="orange_skin_additive",
    starting_num_type="1",
    synthetic_data_size=1000,
    synthetic_data_feature_dim=10,
    pert_num_steps=500,
    save_and_sample_every=100,
    num_workers=4,
)

slurm = SlurmConfig(
    mode="slurm",
    cpus_per_task=4,
    node_list="laniakea",
    slurm_output_folder=f"{output_path}/slurm",
)

subset_sampling_subset_sampler_pert = PertModuleConfig(
    model_type="subset_sampling",
    perturbation_num=10,
    subset_sampling=SubsetSamplingPertModuleConfig(
        use_scheduler=False,
        lr=1e-3,
        tau=0.1,
        tau_start=2,
        tau_anneal_steps=3000,
        tau_scheduler_name="linear",
        hard=False,
        sampler_pert_type="parameterized_subset_sampling",
        sampler_type="subset",
        parameterized_embedding_dim=16,
        parameterized_hidden_dim=32,
        optimizer_name="Adam",
    ),
)

sage_pert = PertModuleConfig(
    model_type="sage",
    perturbation_num=10,
    sage=SAGEPertModuleConfig(
        batch_size=128, n_permutations=25600, detect_convergence=False
    ),
)

lime_pert = PertModuleConfig(
    model_type="lime",
    perturbation_num=10,
    lime=LIMEPertModuleConfig(
        mask_type="random",
        neighbor_size=1024,
        neighbor_feat_num=120000,
        feature_selection="auto",
        distance_metric="cosine",
    ),
)

cxplain_pert = PertModuleConfig(
    model_type="cxplain",
    perturbation_num=10,
    cxplain=CXPlainPertModuleConfig(
        mask_type="sequence",
        batch_size=128,
        trainer_config=CXPlainTrainerConfig(
            model_type="mlp",
            batch_size=128,
            epoch=500,
            early_stopping_patience=10,
            num_layers=4,
            num_units=100,
            learning_rate=5e-4,
            downsample_factors=2,
        ),
    ),
)

fimap_pert = PertModuleConfig(
    model_type="fimap",
    perturbation_num=10,
    fimap=FIMAPPertModuleConfig(
        tau=0.5,
        regularization_weight=1e-1,
        lr=1e-3,
        optimizer_name="Adam",
    ),
)

permutation_pert = PertModuleConfig(
    model_type="permutation",
    perturbation_num=10,
    permutation=PermutationPertModuleConfig(
        trigger_pert_type="trigger_perturbation",
    ),
)

saliency_pert = PertModuleConfig(
    model_type="saliency",
    perturbation_num=10,
    saliency=SaliencyPertModuleConfig(
        smooth_input=False, trigger_pert_type="trigger_perturbation"
    ),
)

smooth_saliency_pert = PertModuleConfig(
    model_type="saliency",
    perturbation_num=10,
    saliency=SaliencyPertModuleConfig(
        smooth_input=True, smooth_number=64, trigger_pert_type="trigger_perturbation"
    ),
)

approximation_pert = PertModuleConfig(
    model_type="approximation",
    perturbation_num=10,
    saliency=ApproximationPertModuleConfig(
        smooth_input=False, trigger_pert_type="trigger_perturbation"
    ),
)

permutation_test_pert = PertModuleConfig(
    model_type="permutation",
    perturbation_num=10,
    permutation=PermutationPertModuleConfig(
        trigger_pert_type="trigger_perturbation",
        strategy="permutation",
        permutation_iter_num=16,
    ),
)

mean_importance_pert = PertModuleConfig(
    model_type="permutation",
    perturbation_num=10,
    permutation=PermutationPertModuleConfig(
        trigger_pert_type="trigger_perturbation",
        strategy="mean",
    ),
)

feature_ablation_pert = PertModuleConfig(
    model_type="permutation",
    perturbation_num=10,
    permutation=PermutationPertModuleConfig(
        trigger_pert_type="trigger_perturbation",
        strategy="ablation",
    ),
)
