Setting up the environment
==========================

```
conda create --name meta-learn-prompt-sanitycheck -y python=3.9 ipython
conda activate meta-learn-prompt-sanitycheck
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt --use-deprecated=legacy-resolver
```

Running
=======

```
tango run rte.jsonnet --include-package catwalk.steps --include-package training_data.py -w workspace
```
