# Adapted from https://github.com/rhewett/distdl-examples/blob/master/lenet5/distributed_experiment.py
# ---
import sys

import torch
from dist_data_handler import get_data_loaders
from mpi4py import MPI
from network import gen_dist_net

Unet_dist = gen_dist_net()

max_batch_size = 256

n_epochs = 10

loud = True

P_base = Unet_dist.P_base

MPI.COMM_WORLD.Barrier() 

parameters = [p for p in Unet_dist.parameters()]
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(parameters,lr=0.0001)

if P_base. rank == 0:
	training_loader, test_loader = get_data_loaders(max_batch_size,
							download=False,
							dummy=False)
else: 
	training_loader, test_loader = get_data_loaders(max_batch_size,
							download=False,
							dummy=True)

# Adapted from https://github.com/activatedgeek/LeNet-5/blob/master/run.py

loss_list, batch_list = [], []

tt = MPI.Wtime()
for epoch in range(n_epochs):
    tte = MPI.Wtime()

    for i, (images, labels) in enumerate(training_loader):

        optimizer.zero_grad()

        # Currently, the exchange algorithm cannot handle varied size
        # tensor inputs. So we must give each pass through the network the same
        # amount of data each time
        if (len(labels) < max_batch_size):
            break

        output = Unet_dist(images)

        # Compute the loss after forward prop. For now, we do this on rank 0 because
        # CrossEntropy is nonlinear, meaning it cannot be simply sumreduced across
        # all ranks to acheive the correct results.
        if P_base.rank == 0:
            loss = criterion(output, labels)
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i + 1)
            if loud and i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss_list[-1]}')
                sys.stdout.flush()
        else:
            loss = output.clone()

        loss.backward()
        optimizer.step()

    if P_base.rank == 0:
        print(f"Epoch {epoch} time: {MPI.Wtime() - tte}")

if P_base.rank == 0:
    print(f"Total time: {MPI.Wtime() - tt}")


total = 0
total_correct = 0
avg_loss = 0.0

for i, (images, labels) in enumerate(test_loader):

    # Currently, the exchange algorithm cannot handle varied size
    # tensor inputs. So we must give each pass through the network the same
    # amount of data each time
    # We do the same in the sequential case to be a fair comparison
    if len(labels) < max_batch_size:
        break

    total += max_batch_size
    output = Unet_dist(images)
    if P_base.rank == 0:
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

perc = float(total_correct) / float(total)

P_base.print_sequential(f"Rank {P_base.rank}, Total Correct: {total_correct}, Total: {total}, %: {perc}")
