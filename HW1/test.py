import torch
print("version: ", torch.__version__)
print("cuda: ",torch.version.cuda)
print("cudnn: ", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())