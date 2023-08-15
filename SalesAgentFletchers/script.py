import torch
import site
import os
print(site.getsitepackages())


print(os.environ.get('CUDA_PATH'))
print(os.environ.get('PATH'))


print(torch.version.cuda)