from pathlib import Path
import torch
import math

RELATIVE_ROOT = Path("..")
# RELATIVE_ROOT = Path("../interpreting-deep-codes")

DATA_DIR = RELATIVE_ROOT / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOGS_DIR = RELATIVE_ROOT / "logs"
FIGS_DIR = RELATIVE_ROOT / "figs"

TMP_DIR = RELATIVE_ROOT / "tmp"

EXPERIMENT_SETTINGS_JSON = RELATIVE_ROOT / "experiment_settings.json"

MODELS_DIR = RELATIVE_ROOT / "models"
CHECKPOINTS_DIR = RELATIVE_ROOT / "checkpoints"
TURBOAE_DECODER_BINARY_PATH = MODELS_DIR / "turboae_binary_decoder.pt"
TURBOAE_ENCODER_BINARY_PATH = MODELS_DIR / "turboae_binary_encoder.pt"
TURBOAE_ENCODER_CONT_PATH = MODELS_DIR / "turboae_cont_encoder.pt"
TURBOAE_DECODER_CONT_PATH = MODELS_DIR / "turboae_cont_decoder.pt"

TURBOAE_ENCODER_CONT_MEAN = 0.27577412655949557
TURBOAE_ENCODER_CONT_STD = math.sqrt(0.2694892215728756)

INPUT_SYMBOL = "u"

TURBOAE_EXACT_TABLE1_BITS_3_98 = torch.CharTensor(
    [
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
    ],
)
TURBOAE_EXACT_TABLE2_BITS_3_98 = torch.CharTensor(
    [
        [1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [1],
        [0],
        [0],
        [1],
        [1],
        [0],
        [0],
        [1],
    ],
)

TURBOAE_INTERLEAVER_PERMUTATION = torch.LongTensor(
    [
        26,
        86,
        2,
        55,
        75,
        93,
        16,
        73,
        54,
        95,
        53,
        92,
        78,
        13,
        7,
        30,
        22,
        24,
        33,
        8,
        43,
        62,
        3,
        71,
        45,
        48,
        6,
        99,
        82,
        76,
        60,
        80,
        90,
        68,
        51,
        27,
        18,
        56,
        63,
        74,
        1,
        61,
        42,
        41,
        4,
        15,
        17,
        40,
        38,
        5,
        91,
        59,
        0,
        34,
        28,
        50,
        11,
        35,
        23,
        52,
        10,
        31,
        66,
        57,
        79,
        85,
        32,
        84,
        14,
        89,
        19,
        29,
        49,
        97,
        98,
        69,
        20,
        94,
        72,
        77,
        25,
        37,
        81,
        46,
        39,
        65,
        58,
        12,
        88,
        70,
        87,
        36,
        21,
        83,
        9,
        96,
        67,
        64,
        47,
        44,
    ],
)