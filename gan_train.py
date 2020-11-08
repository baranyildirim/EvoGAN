from typing import List
from AutoGAN.train_derived import train_derived
#from munch import Munch
import AutoGAN.cfg

args = dict([
    ("-gen_bs", 128),
    ("-dis_bs", 64),
    ("--dataset", "cifar10"),
    ("--bottom_width", 4),
    ("--img_size", 32),
    ("--max_iter", 50000),
    ("--gen_model", "shared_gan"),
    ("--dis_model", "shared_gan"),
    ("--latent_dim", 128),
    ("--gf_dim", 256),
    ("--df_dim", 128),
    ("--g_spectral_norm", False),
    ("--d_spectral_norm", True),
    ("--g_lr", 0.0002),
    ("--d_lr", 0.0002),
    ("--beta1", 0.0),
    ("--beta2", 0.9),
    ("--init_type", "xavier_uniform"),
    ("--n_critic", 5),
    ("--val_freq", 20),
    ("--exp_name", "derive"),
])

def train_gan(arch: List[int], max_epoch: int) -> float:
    args["--arch"] = arch
    args["--max_epoch"] = max_epoch
    args_string = ", ".join(f'{x[0]}={x[1]!r}' for x in args.items())
    return train_derived(AutoGAN.cfg.parse_args(args=args_string))   