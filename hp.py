from constants import ConstDict


class HyperParams:
    model_name = "efficient_split_two"
    save_folder = "saved_models"
    seed = 420

    # Data

    # HDF
    using_hdf = False
    hdf_key = "audio"

    # Splits
    using_split = True
    split_csv = "/home/teh_devs/deepfake/dataset/fake-real-distinct.csv"
    per = 5000
    train_idx_list = [(1, 1, 1)]
    test_idx_list = [(2, 2, 1)]
    split_seed = 50
    shuffle_fake = False
    shuffle_fake_seed = 50

    data_dir = "/home/teh_devs/deepfake/dataset/revamp"
    using_augments = False
    use_pinned_memory_train = True
    use_pinned_memory_test = True
    test_batch_size = 40
    train_batch_size = 40
    data_num_workers = 30
    test_split_percent = 0.1
    balanced_sampling = True

    # Train
    num_epochs = 10

    # Optimizer
    weight_decay = 0.001
    lr = 0.00005
    betas = (0.9, 0.999)
    amsgrad = False

    # Criterion
    use_class_weights = False
    class_weights = [1, 2]

    # Scheduler
    use_step_lr = False
    step_sched_params = {"step_size": 10, "gamma": 0.1, "last_epoch": -1}

    use_plateau_lr = False
    plateau_lr_sched_params = {
        "mode": "min",  # Passing in epoch loss, keep it min
        "patience": 3,  # Num epochs to ignore before reducing
        "factor": 0.5,  # How much to redue lr by
        "verbose": True,
        "threshold": 0.0001,  # Number of decimal places to consider when reducing
        "threshold_mode": "rel",
        "cooldown": 10,  # After reducing, how many epochs to wait before start monitoring again
        "min_lr": 0,
        "eps": 1e-08,  # If newlr - oldlr < eps, update is ignored
    }

    use_one_cycle_lr = True
    oc_sched_params = {
        "max_lr": 0.00005,
        "div_factor": 100.0,  # initial_lr = max_lr/div_factor
        "final_div_factor": 10000.0,  # min_lr = initial_lr/final_div_factor
        "epochs": num_epochs,
        "pct_start": 0.30,  # percentage of time going up/down
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "last_epoch": -1,  # Change this if resuming (Pass in total number of batches done, not epochs!!)
    }


hp = ConstDict(
    **{key: val for key, val in vars(HyperParams).items() if not key.startswith("__")}
)
