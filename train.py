import argparse
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

from nfnet import NFNet, nfnet_params
from dataset import makeDataset
import os

LABEL_SMOOTHING=0.1
LEARNING_RATE=0.001
DROPRATE=0.2
EMA_DECAY=0.99999
CLIPPING_FACTOR=0.01



def parse_args():
    ap = argparse.ArgumentParser()

    #data_path
    ap.add_argument(
        "-dp",
        "--data_path",
        type=str,
        help="training data path",
        required=True
    )

    #model_path
    ap.add_argument(
        "-mp",
        "--model_path",
        type=str,
        help="model path",
    )

    ap.add_argument(
        "-v",
        "--variant",
        default="F0",
        type=str,
        help="model variant",
    )
    ap.add_argument(
        "-b",
        "--batch_size",
        default=16,
        type=int,
        help="train batch size",
    )
    ap.add_argument(
        "-n",
        "--num_epochs",
        default=300,
        type=int,
        help="number of training epochs",
    )
    
    return ap.parse_args()


def main(args):

    try:
        ngFileNames=os.listdir(os.path.join(args.data_path,"train","0"))
        okFileNames=os.listdir(os.path.join(args.data_path,"train","1"))
        numberOfImage=len(ngFileNames)+len(okFileNames)
        
        steps_per_epoch = numberOfImage / args.batch_size
        training_steps = (numberOfImage * args.num_epochs) / args.batch_size
        train_imsize = nfnet_params[args.variant]["train_imsize"]
        test_imsize = nfnet_params[args.variant]["test_imsize"]

        
        
        eval_preproc = "resize_crop_32"

        model = NFNet(
            num_classes=1000,
            variant=args.variant,
            drop_rate=DROPRATE,
            label_smoothing=LABEL_SMOOTHING,
            ema_decay=EMA_DECAY,
            clipping_factor=CLIPPING_FACTOR
        )

        model.build((1, train_imsize, train_imsize, 3))  #batch_input_shape

        max_lr = LEARNING_RATE * args.batch_size / 256
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=max_lr,
            decay_steps=training_steps - 5 * steps_per_epoch,
        )

        lr_schedule = WarmUp(
            initial_learning_rate=max_lr,
            decay_schedule_fn=lr_decayed_fn,
            warmup_steps=5 * steps_per_epoch,
        )

        optimizer = tfa.optimizers.SGDW(
            learning_rate=lr_schedule, weight_decay=2e-5, momentum=0.9
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="top_1_acc"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(
                    k=5, name="top_5_acc"
                ),
            ],
        )

        ds_train,ds_valid=makeDataset(dataPath=args.data_path,batch_size=args.batch_size)

        model.fit(#TODO 改成dataset pipeline
            ds_train,
            validation_data=ds_test,
            epochs=args.num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[tf.keras.callbacks.TensorBoard()],
        )
    except Exception as e:
        print(e)

# Patched from: https://huggingface.co/transformers/_modules/transformers/optimization_tf.html#WarmUp
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.
    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


if __name__ == "__main__":
    args = parse_args()
    main(args)