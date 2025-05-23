import subprocess

subprocess.run("python preprocess.py", shell=True)
subprocess.run("python train.py", shell=True)
subprocess.run("python hpo.py", shell=True)
subprocess.run("python register.py", shell=True)