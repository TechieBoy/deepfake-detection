from constants import ConstDict


class HyperParams:
    model_name = "audio_base_cnn_step_lr_test_convergence"
    save_folder = "saved_models"
    seed = 420

    # Data
    using_hdf = True
    hdf_key = "audio"
    data_dir = 'audio.hdf5'
    using_augments = False
    use_pinned_memory_train = True
    use_pinned_memory_test = True
    test_batch_size = 40
    train_batch_size = 40
    data_num_workers = 30
    test_split_percent = 0.1
    balanced_sampling = True

    # Train
    num_epochs = 20

    # Optimizer
    weight_decay = 0.01
    lr = 0.01

    # Criterion
    use_class_weights = False
    class_weights = [1, 2]

    # Scheduler
    use_step_lr = True
    step_sched_params = {
        "step_size": 10,
        "gamma": 0.1,
        "last_epoch": -1
    }

    use_plateau_lr = False
    plateau_lr_sched_params = {
        "mode": "min",  # Passing in epoch loss, keep it min
        "patience": 10,  # Num epochs to ignore before reducing
        "factor": 0.1,  # How much to redue lr by
        "verbose": False,
        "threshold": 0.0001,  # Number of decimal places to consider when reducing
        "threshold_mode": "rel",
        "cooldown": 0,  # After reducing, how many epochs to wait before start monitoring again
        "min_lr": 0,
        "eps": 1e-08,  # If newlr - oldlr < eps, update is ignored
    }

    use_one_cycle_lr = False
    oc_sched_params = {
        "max_lr": 0.01,
        "div_factor": 20.0,  # initial_lr = max_lr/div_factor
        "final_div_factor": 10000.0,  # min_lr = initial_lr/final_div_factor
        "epochs": num_epochs,
        "pct_start": 0.40,  # percentage of time going up/down
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "last_epoch": -1,  # Change this if resuming (Pass in total number of batches done, not epochs!!)
    }


hp = ConstDict(
    **{key: val for key, val in vars(HyperParams).items() if not key.startswith("__")}
)
