from typing import List
from AutoGAN.train_derived import train_derived
import AutoGAN.cfg

from multiprocessing import Process, Queue

args = dict([
    ("-gen_bs", 128),
    ("-dis_bs", 128),
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
    ("--calc_fid", False),
    ("--warnings_enabled", False),
    ("--num_eval_imgs", 10000),
])

def _train_gan(arch: List[int], max_epoch: int, q):
    args["--max_epoch"] = max_epoch
    args_list = []
    for k, v in args.items():
        args_list.extend([k, str(v)])
    args_list.append("--arch")
    for item in arch:
        args_list.append(str(item))

    result = train_derived(AutoGAN.cfg.parse_args(args=args_list))   
    q.put(result)
    return

def train_gan(arch: List[int], max_epoch: int) -> float:
    queue = Queue()
    p = Process(target=_train_gan, args=(arch, max_epoch, queue))
    p.start()
    p.join()
    result = queue.get()
    return result