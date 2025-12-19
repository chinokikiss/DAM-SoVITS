import os

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import logging
import platform
from pathlib import Path

import time
import shutil
import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt

def my_save(fea, path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    name = os.path.basename(path)
    tmp_path = "%s.pth" % (time.time())
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


def get_dist_strategy_and_devices():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Detected {gpu_count} GPU(s) available.")
        if gpu_count > 1:
            print("Environment supports DDP multi-GPU training, configuring DDP strategy...")
            backend = "gloo" if platform.system() == "Windows" else "nccl"
            strategy = DDPStrategy(
                process_group_backend=backend,
                find_unused_parameters=False
            )
            devices = -1
            return devices, strategy, "gpu"
        
        else:
            print("Only one GPU detected, using single GPU mode with auto strategy.")
            return 1, "auto", "gpu"
    else:
        print("No GPU detected, using CPU mode.")
        return 1, "auto", "cpu"


class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)

                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"
    devices, strategy, accelerator = get_dist_strategy_and_devices()
    trainer: Trainer = Trainer(
        max_epochs=config["train"]["epochs"],
        accelerator=accelerator,
        limit_val_batches=0,
        devices=devices, 
        benchmark=False,
        fast_dev_run=False,
        strategy=strategy, 
        precision=config["train"]["precision"],
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[ckpt_callback],
        use_distributed_sampler=False,
    )

    model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)

    data_module: Text2SemanticDataModule = Text2SemanticDataModule(
        config,
        data_path=config["data_path"],
    )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)
    trainer.fit(model, data_module, ckpt_path=ckpt_path)


# python DAM_SoVITS\s1_train.py -c DAM_SoVITS\configs\s1.yaml
# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1.yaml",
        help="path of config file",
    )

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
