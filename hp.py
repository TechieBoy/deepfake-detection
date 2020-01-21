from constants import ConstDict


class HyperParams:
    model_name = "model_name"
    save_folder = "saved_models"
    seed = 420

    # Data
    using_augments = False
    use_pinned_memory_train = True
    use_pinned_memory_test = True
    data_dir = 'dir_directory'
    test_batch_size = 120
    train_batch_size = 120
    data_num_workers = 30
    test_split_percent = 0.1
    balanced_sampling = True

    # Train
    num_epochs = 20

    # Optimizer
    weight_decay = 0.05
    lr = 0.003

    # Criterion
    use_class_weights = True
    class_weights = [1, 6]

    # Scheduler
    use_one_cycle_lr = True
    oc_sched_params = {
        "max_lr": 0.001,
        "div_factor": 5.0,  # initial_lr = max_lr/div_factor
        "final_div_factor": 10000.0,  # min_lr = initial_lr/final_div_factor
        "epochs": num_epochs,
        # "steps_per_epoch": len(dataloaders["train"]),
        "pct_start": 0.30,  # percentage of time going up/down
        "cycle_momentum": True,
        "base_momentum": 0.85,
        "max_momentum": 0.95,
        "last_epoch": -1,  # Change this if resuming (Pass in total number of batches done, not epochs!!)
    }

    use_step_lr = False
    step_sched_params = {
        "step_size": 10,
        "gamma": 0.1,
        "last_epoch": -1
    }


hp = ConstDict(
    **{key: val for key, val in vars(HyperParams).items() if not key.startswith("__")}
)
