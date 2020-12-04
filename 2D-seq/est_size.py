from network_size_est import Unet

model = Unet()

in_size = (1, 3, 64, 64)

from pytorch_modelsize import SizeEstimator

se = SizeEstimator(model, in_size)

se.estimate_size()

def b_to_GB(x):

	return (x/8)/(1024**3)

print(f"Input data: {b_to_GB(se.input_bits)} GB")
print(f"Foward data: {b_to_GB(se.forward_backward_bits/2)} GB")
print(f"Backward data: {b_to_GB(se.forward_backward_bits/2)} GB")
print(f"Model data: {b_to_GB(se.param_bits)} GB")
print(f"Total data: {b_to_GB(se.total_bits)} GB")
