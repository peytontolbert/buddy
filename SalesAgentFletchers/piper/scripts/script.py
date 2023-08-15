import torch
import site
import os
#print(site.getsitepackages())

#print(os.environ.get('CUDA_PATH'))
#print(os.environ.get('PATH'))

#print(torch.version.cuda)

os.system("echo 'Welcome to the world of speech synthesis!' | piper --model en_US-lessac-medium --output_file welcome.wav")