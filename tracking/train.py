import os
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_TRAINING = os.path.join(ROOT_DIR, "lib", "train", "run_training.py")
DEFAULT_SCRIPT = "ostrack_st1_quadra_zfusetrans_3ioubranch_rgbtpos_regwcls_3head_specific2anotherall_wsq"
DEFAULT_CONFIG = "vitb_256_ostrack_stark_32x1_1e4_lasher_30ep_sot_11_wmul0p1_bat"
DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "output", DEFAULT_CONFIG)

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train wmul10记得改回来
    parser.add_argument('--script', type=str, default=DEFAULT_SCRIPT, help='training script name')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG, help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multi_node"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, default=2, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0)  # whether datasets are in lmdb format
    parser.add_argument('--script_prv', type=str, default='ostrack_base_model_specific_template_concatattfusion', help='training script name')
    parser.add_argument('--config_prv', type=str, default='vitb_256_ostrack_stark_32x1_1e4_lasher_30ep_sot_11', help='yaml configure file name')
    parser.add_argument('--use_wandb', type=int, choices=[0, 1], default=0)  # whether to use wandb
    # for knowledge distillation
    parser.add_argument('--distill', type=int, choices=[0, 1], default=0)  # whether to use knowledge distillation
    parser.add_argument('--script_teacher', type=str, help='teacher script name')
    parser.add_argument('--config_teacher', type=str, help='teacher yaml configure file name')

    # for multiple machines
    parser.add_argument('--rank', type=int, help='Rank of the current process.')
    parser.add_argument('--world-size', type=int, help='Number of processes participating in the job.')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
    parser.add_argument('--port', type=int, default='20000', help='Port of the current rank 0.')

    parser.add_argument('--vis_gpus', type=str, default='6,7')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.vis_gpus

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = "python %s --script %s --config %s --save_dir %s --use_lmdb %d " \
                    "--script_prv %s --config_prv %s --distill %d --script_teacher %s --config_teacher %s --use_wandb %d"\
                    % (RUN_TRAINING, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv,
                       args.distill, args.script_teacher, args.config_teacher, args.use_wandb)
    elif args.mode == "multiple":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_port %d %s " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, random.randint(10000, 50000), RUN_TRAINING, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    elif args.mode == "multi_node":
        train_cmd = "python -m torch.distributed.launch --nproc_per_node %d --master_addr %s --master_port %d --nnodes %d --node_rank %d lib/train/run_training.py " \
                    "--script %s --config %s --save_dir %s --use_lmdb %d --script_prv %s --config_prv %s --use_wandb %d " \
                    "--distill %d --script_teacher %s --config_teacher %s" \
                    % (args.nproc_per_node, args.ip, args.port, args.world_size, args.rank, args.script, args.config, args.save_dir, args.use_lmdb, args.script_prv, args.config_prv, args.use_wandb,
                       args.distill, args.script_teacher, args.config_teacher)
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
