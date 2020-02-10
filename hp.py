from constants import ConstDict


class HyperParams:
    model_name = "efficient_revamp_weighted_with_transform"
    save_folder = "saved_models"
    seed = 420

    # Data

    # HDF
    using_hdf = False
    hdf_key = "audio"

    # Splits
    using_split = True
    split_csv = "/home/bcd/deepfake/dataset/fake-real-distinct.csv"
    per = 5000
    train_idx_list = [(0, 0, 0)]
    test_idx_list = [(1, 1, 0)]
    split_seed = 50
    shuffle_fake = False
    shuffle_fake_seed = 50

    # Default
    data_dir = "/raid/deepfake/revamp"
    using_augments = False
    balanced_sampling = False

    # FWA data
    using_fwa = False
    real_folder_loc = "/raid/deepfake/revamp"
    fake_loc = "/raid/deepfake/finale"

    # Common data
    use_pinned_memory_train = True
    use_pinned_memory_test = True
    test_batch_size =  256
    train_batch_size = 256
    data_num_workers = 24
    test_split_percent = 0.1

    # Train
    num_epochs = 32

    # Optimizer
    use_adamW = False
    adamW_params = dict(weight_decay=0.001, lr=2e-4, betas=(0.9, 0.999), amsgrad=False)

    use_sgd = True
    sgd_params = dict(
        lr=1e-3, momentum=0.9, dampening=0, weight_decay=0.01, nesterov=True
    )

    # Criterion
    use_class_weights = True
    class_weights = [1.1, 1]

    # Scheduler
    use_step_lr = False
    step_sched_params = {"step_size": 2, "gamma": 0.95, "last_epoch": -1}

    use_plateau_lr = False
    plateau_lr_sched_params = {
        "mode": "min",  # Passing in epoch loss, keep it min
        "patience": 1,  # Num epochs to ignore before reducing
        "factor": 0.95,  # How much to redue lr by
        "verbose": True,
        "threshold": 0.0001,  # Number of decimal places to consider when reducing
        "threshold_mode": "rel",
        "cooldown": 1,  # After reducing, how many epochs to wait before start monitoring again
        "min_lr": 0,
        "eps": 1e-08,  # If newlr - oldlr < eps, update is ignored
    }

    use_one_cycle_lr = True
    oc_sched_params = {
        "max_lr": 1e-3,
        "div_factor": 10.0,  # initial_lr = max_lr/div_factor
        "final_div_factor": 1000.0,  # min_lr = initial_lr/final_div_factor
        "epochs": num_epochs,
        "pct_start": 0.42,  # percentage of time going up/down
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.9,
        "last_epoch": -1,  # Change this if resuming (Pass in total number of batches done, not epochs!!)
    }

    use_cos_anneal_restart = False
    cos_anneal_sched_params = {
        "eta_min": 0.0017,  # Minimum Learning rate
        "last_epoch": -1,
    }


hp = ConstDict(
    **{key: val for key, val in vars(HyperParams).items() if not key.startswith("__")}
)
