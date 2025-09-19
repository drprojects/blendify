Here are my instructions for installing what I need. I found that 
installing the `blendify` dependencies was quite hard.

```bash
# Make sure you have a recent conda>=22
# https://github.com/ptrvilya/blendify/issues/6
conda create --name blendify python=3.10.9

conda activate blendify

pip install bpy==3.5.0
pip install blendify
pip install torch
pip install blender_plots
pip install trimesh
pip install videoio
pip install matplotlib
pip install mathutils
```
