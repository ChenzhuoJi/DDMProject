import ml_collections

def get_config():

    cifar10_path = 'path/to/cifar10'

    config = ml_collections.ConfigDict()
    config.eval_name = 'CIFAR_elbo'
    config.save_location = 'path/to/save_location'
    config.train_config_overrides = [
        [['device'], 'cuda'],
        [['data', 'root'], cifar10_path],
        [['distributed'], False]
    ]
    config.experiment_dir = 'path/to/experiment_dir'
    config.checkpoint_path = 'path/to/experiment_dir/checkpoints/ckpt_0001999999.pt'

    config.loggers = ['ELBO']

    config.device = 'cuda'

    config.data = data = ml_collections.ConfigDict()
    data.name = 'DiscreteCIFAR10'
    data.root = cifar10_path
    data.train = False
    data.download = True
    data.S = 256
    data.batch_size = 16
    data.shuffle = True
    data.shape = [3,32,32]
    data.random_flips = False

    config.sampler = sampler = ml_collections.ConfigDict()
    sampler.name = 'TauLeaping'
    sampler.num_steps = 1000
    sampler.min_t = 0.02
    sampler.eps_ratio = 1e-9
    sampler.finish_strat = 'max'
    sampler.theta = 1.0
    sampler.initial_dist = 'gaussian'
    sampler.num_corrector_steps = 1
    sampler.corrector_step_size_multiplier = 1.0
    sampler.corrector_entry_time = 1.0

    config.logging = logging = ml_collections.ConfigDict()
    logging.total_N = 100
    logging.total_B = 10000
    logging.B = 50
    logging.min_t = 0.01
    logging.eps = 1e-9
    logging.initial_dist = 'gaussian'

    return config