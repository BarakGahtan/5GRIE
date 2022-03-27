def init_constants(opts):
    HP = dict(
        eps_eval = opts.episode_eval,
        eps_compare = opts.episode_compare,
        centralized_agent = opts.center_agent,
        learn_rate=opts.learning_rate,
        eps_max=opts.epsilon,  # exploration probability at start
        eps_min=0.01,  # minimum exploration probability
        epsilon_decay=opts.eps_decay,  # exponential decay rate for exploration prob
        batch_size=opts.batch_size,
        gamma=0.98,  # discount rate
        alpha=0.2,
        buffer_size=opts.buffer_size,
        tau=0.1,  # target network soft update hyper parameters
        K = opts.transceiver_count,  # number of transceivers
        max_steps_per_episode=opts.max_step_per_episode,
        rings=opts.rings_count,
    )
    return HP
