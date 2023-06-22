class AgentConfig:
    # Learning
    gamma = 1
    train_freq = 1
    start_learning = 10
    memory_size = 3000
    batch_size = 32
    plot_every = 50
    reset_step = 10

    epsilon = 1
    epsilon_minimum = 0.03
    epsilon_decay_rate = 0.99993
    learning_rate = 0.0005

    max_step = 40000000       # 40M steps max
    max_episode_length = 500  # equivalent of 5 minutes of game play at 60 frames per second

    # Algorithm selection
    train_cartpole = True
    per = False

    double_q_learning = False
    duelling_dqn = False

    gif = False
    gif_every = 9999999
    
    # 
    obs_size = 4
    latent_size = 8
    n_layers = 1
    enc_dropout = 0
    dec_dropout = 0
    learning_rate_encd = 0.0001


class EnvConfig:
    env_name = 'CartPole-v0'
    save_every = 10000
