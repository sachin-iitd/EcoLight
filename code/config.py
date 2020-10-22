# parameters and paths
from dqn import DQN
from anon_env import AnonEnv

DIC_PHASE_MAP_CYCLIC = {
    0: 5,
    1: 7,
    2: 6,
    3: 8,
    -1: 0
}

DIC_EXP_CONF = {
    "RUN_COUNTS": 3600,
    "TRAFFIC_FILE": None,
    "MODEL_NAME": "DQN",
    "START_ROUNDS": 0,
    "NUM_ROUNDS": 300,
    "NUM_GENERATORS": 1,
    "LIST_MODEL": ["DQN"],
    "LIST_MODEL_NEED_TO_UPDATE": ["DQN"],
}

dic_traffic_env_conf = {
    "ACTION_PATTERN": 1, # 0=set, 1=switch-next
    "NUM_INTERSECTIONS": 1,
    "MIN_ACTION_TIME": 10,
    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "MEASURE_TIME": 10,
    "INTERVAL": 1,
    "SAVEREPLAY": False,
    "RLTRAFFICLIGHT": True,
    "SINGLE_AGENT": False,
    "TRANSFER": False,
    "DIC_PHASE_MAP": DIC_PHASE_MAP_CYCLIC,

    "DIC_FEATURE_DIM": dict(
        D_TRANSFORM_2DIM=(2,),
        D_TRANSFORM_1DIM=(1,),
    ),

    "LIST_STATE_FEATURE": ['transform_2dim'],
    "DIC_REWARD_INFO": {'stop': -0.25},

    "LANE_NUM": {
        "LEFT": 1,
        "RIGHT": 1,
        "STRAIGHT": 1
    }
}

DIC_DQN_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "UPDATE_Q_BAR_FREQ": 5,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": [20],
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
    "QSKIP": 0,
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
}

DIC_AGENTS = {
    "DQN": DQN,
}

DIC_ENVS = {
    "anon": AnonEnv
}
