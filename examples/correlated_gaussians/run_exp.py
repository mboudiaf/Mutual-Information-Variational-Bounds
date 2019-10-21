import subprocess

subprocess.run(["python3", "demo_script_gaussian.py",
                "--regularizer", "mine",
                "--data_size", "100000",
                "--epochs", "10"
                ])

subprocess.run(["python3", "demo_script_gaussian.py",
                "--regularizer", "nwj",
                "--data_size", "100000",
                "--epochs", "10"
                ])

subprocess.run(["python3", "demo_script_gaussian.py",
                "--regularizer", "nce",
                "--data_size", "100000",
                "--epochs", "10"
                ])