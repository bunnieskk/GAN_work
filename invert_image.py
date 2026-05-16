#!/usr/bin/env python3
"""Image inversion for NVlabs StyleGAN (TensorFlow 1.x).

This script optimizes W (or W+) so that the generated image matches a target photo.
It is designed for this repository layout and pre-trained pickles from README.
"""

import argparse
import os
import pickle

import numpy as np
from PIL import Image
import tensorflow as tf

import config
import dnnlib
import dnnlib.tflib as tflib


DEFAULT_NETWORK = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'


def _load_target(path, resolution):
    img = Image.open(path).convert('RGB')
    img = img.resize((resolution, resolution), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = arr[np.newaxis, ...]  # NCHW
    return arr


def _save_nchw_uint8(path, images):
    arr = images[0]
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr.astype(np.uint8), 'RGB').save(path)


def invert(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('input', exist_ok=True)

    tflib.init_tf()

    with dnnlib.util.open_url(args.network, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    resolution = Gs.output_shape[2]
    target = _load_target(args.input, resolution)

    dlatent_avg = Gs.get_var('dlatent_avg')  # [512]
    synth = Gs.components.synthesis
    num_layers = synth.input_shape[1]
    dlatent_size = synth.input_shape[2]

    with tf.name_scope('inversion'):
        target_ph = tf.constant(target, dtype=tf.uint8)
        target_f = tf.cast(target_ph, tf.float32)

        if args.space == 'wplus':
            w_init = np.tile(dlatent_avg[np.newaxis, np.newaxis, :], [1, num_layers, 1])
        else:
            w_init = dlatent_avg[np.newaxis, np.newaxis, :]

        w_var = tf.Variable(w_init.astype(np.float32), name='w_opt')

        if args.space == 'w':
            dlatents = tf.tile(w_var, [1, num_layers, 1])
        else:
            dlatents = w_var

        gen = synth.get_output_for(dlatents, randomize_noise=False, is_validation=True)
        gen_uint8 = tflib.convert_images_to_uint8(gen)
        gen_f = tf.cast(gen_uint8, tf.float32)

        mse = tf.reduce_mean(tf.square(gen_f - target_f))
        w_anchor = tf.constant(dlatent_avg[np.newaxis, np.newaxis, :], dtype=tf.float32)
        reg = tf.reduce_mean(tf.square(dlatents - w_anchor))
        loss = mse + args.lambda_reg * reg

        lr = tf.Variable(args.lr, trainable=False, dtype=tf.float32, name='lr')
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(loss, var_list=[w_var])

    tflib.init_uninitialized_vars()

    for step in range(1, args.steps + 1):
        loss_val, mse_val, reg_val, _ = tflib.run([loss, mse, reg, train_op])
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"step {step:5d}/{args.steps}  loss={loss_val:.4f}  mse={mse_val:.4f}  reg={reg_val:.6f}")

    out_img, out_w = tflib.run([gen_uint8, dlatents])

    stem = os.path.splitext(os.path.basename(args.input))[0]
    img_path = os.path.join(args.output_dir, f'{stem}_recon.png')
    w_path = os.path.join(args.output_dir, f'{stem}_{args.space}.npy')
    _save_nchw_uint8(img_path, out_img)
    np.save(w_path, out_w)

    print('saved:', img_path)
    print('saved:', w_path)


def build_argparser():
    p = argparse.ArgumentParser(description='Invert a real photo into StyleGAN latent space (W/W+).')
    p.add_argument('--input', required=True, help='Path to input image (recommend putting it under ./input).')
    p.add_argument('--output-dir', default='results/inversion', help='Output directory.')
    p.add_argument('--network', default=DEFAULT_NETWORK, help='Pretrained network pickle URL/path.')
    p.add_argument('--space', choices=['w', 'wplus'], default='wplus', help='Optimize in W or W+ space.')
    p.add_argument('--steps', type=int, default=400, help='Optimization steps.')
    p.add_argument('--lr', type=float, default=0.05, help='Adam learning rate.')
    p.add_argument('--lambda-reg', type=float, default=1e-4, help='Regularization weight toward dlatent_avg.')
    p.add_argument('--log-every', type=int, default=20, help='Log frequency.')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    invert(args)
