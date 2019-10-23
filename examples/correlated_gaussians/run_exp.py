#!/usr/bin/env python

"""run_exp.py: Script to execute in order to produce experiments."""

import subprocess

subprocess.run(["python3", "demo_gaussian.py",
                "--regularizer", "nce",
                "--critic_type", "joint",
                "--data_size", "100000",
                "--epochs", "10"
                ])


subprocess.run(["python3", "demo_gaussian.py",
                "--regularizer", "mine",
                "--critic_type", "separate",
                "--ema_decay", "0.9",
                "--data_size", "100000",
                "--epochs", "10"
                ])

subprocess.run(["python3", "demo_gaussian.py",
                "--regularizer", "nwj",
                "--critic_type", "separate",
                "--data_size", "100000",
                "--epochs", "10"
                ])




