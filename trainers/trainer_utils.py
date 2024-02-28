import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn


def merge_results_dist(tmpdir, part_logits, part_labels):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(torch.cat(part_logits).cpu().numpy(),
                open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    pickle.dump(torch.cat(part_labels).cpu().numpy(),
                open(os.path.join(tmpdir, 'label_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank == 0:

        part_list = []
        part_label_list = []

        for i in range(world_size):
            part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
            part_label = os.path.join(tmpdir, 'label_part_{}.pkl'.format(i))

            part_list.append(pickle.load(open(part_file, 'rb')))
            part_label_list.append(pickle.load(open(part_label, 'rb')))

        part_list = np.concatenate(part_list, axis=0)
        part_label_list = np.concatenate(part_label_list, axis=0)

        logits_all = torch.from_numpy(part_list)
        labels_all = torch.from_numpy(part_label_list)

        return logits_all, labels_all
    else:
        return None, None


def merge_two_branch_results_dist(tmpdir, part_image_logits, part_text_logits, part_labels):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(torch.cat(part_image_logits).cpu().numpy(),
                open(os.path.join(tmpdir, 'result_part_image_{}.pkl'.format(rank)), 'wb'))
    pickle.dump(torch.cat(part_text_logits).cpu().numpy(),
                open(os.path.join(tmpdir, 'result_part_text_{}.pkl'.format(rank)), 'wb'))
    pickle.dump(torch.cat(part_labels).cpu().numpy(),
                open(os.path.join(tmpdir, 'label_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank == 0:

        part_image_list = []
        part_text_list = []
        part_label_list = []

        for i in range(world_size):
            part_image_file = os.path.join(tmpdir, 'result_part_image_{}.pkl'.format(i))
            part_text_file = os.path.join(tmpdir, 'result_part_text_{}.pkl'.format(i))
            part_label = os.path.join(tmpdir, 'label_part_{}.pkl'.format(i))

            part_image_list.append(pickle.load(open(part_image_file, 'rb')))
            part_text_list.append(pickle.load(open(part_text_file, 'rb')))
            part_label_list.append(pickle.load(open(part_label, 'rb')))

        part_image_list = np.concatenate(part_image_list, axis=0)
        part_text_list = np.concatenate(part_text_list, axis=0)
        part_label_list = np.concatenate(part_label_list, axis=0)

        logits_image = torch.from_numpy(part_image_list)
        logits_text = torch.from_numpy(part_text_list)
        labels_all = torch.from_numpy(part_label_list)

        return logits_image, logits_text, labels_all
    else:
        return None, None, None