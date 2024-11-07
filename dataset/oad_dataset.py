import torch.utils.data as data
import os
import pickle
import numpy as np
import torch
import random

class ActionDataset(data.Dataset):
    def __init__(self, config, split):
        random.seed(config.seed)
        self.img_tmpl = config.data.img_tmpl
        self.index_bias = config.data.index_bias
        self.downsample = config.data.downsample
        self.dataset = config.data.dataset
        self.data_path = os.path.join(config.data.data_path, self.dataset, 'FrameEmbeddings-vitb16')
        self.anno_path = os.path.join(config.data.anno_path, self.dataset)
        self.split = split
        self.num_segments = config.data.num_segments
        self.fut_anticipation = config.data.fut_anticipation
        self.future_steps = config.data.future_steps

        print('Loading {} split of the {} dataset from {}'.format(self.split, self.dataset, self.data_path))

        # load classes from file
        with open(os.path.join(self.anno_path, 'classes.txt'), 'r') as f:
            self.classes = f.read().splitlines()


        # load annotations from pickle file
        with open(os.path.join(self.anno_path, split + '.pickle'), 'rb') as f:
            annotations = pickle.load(f)

        self._parse_annotations(annotations)

        if config.data.few_shot and split == 'train':
            self.few_shot_inputs(config.data.few_shot)

    def _parse_annotations(self, annotations):
        self.inputs = []
        background_inputs = []
        for video in annotations.keys():
            num_fts = annotations[video]['feature_length']
            start_offset = annotations[video]['start_offset']
            end_offset = annotations[video]['end_offset']

            for start, end in zip(range(start_offset, end_offset - (self.num_segments + self.future_steps), 1),
                                range(start_offset + self.num_segments, end_offset - self.future_steps, 1)):
                
                target = annotations[video]['anno'][end-1]
                if np.sum(target) == 0: # ambigous class in thumos dataset
                    continue
                target = np.argmax(target)
                

                future_targets = -1
                if self.future_steps > 0:
                    future_targets = annotations[video]['anno'][end:end+self.future_steps]
                    future_targets = future_targets[0]
                    if np.sum(future_targets) == 0: # ambigous class in thumos dataset
                        continue
                    future_targets = np.argmax(future_targets)


                if target == 0:
                    if self.split == 'train': # Background frames are not evaluated (same result, faster evaluation)
                        background_inputs.append([video, start, end, target, future_targets])
                else:
                    self.inputs.append([video, start, end, target, future_targets])
        
        if self.split == 'train':
            # Number of samples per class
            samples_per_class = np.zeros(len(self.classes))
            for _, _, _, target, _ in self.inputs:
                samples_per_class[target] += 1

            # Get class with most samples without background
            max_samples = np.max(samples_per_class[1:]).astype(int)

            # Generate max_samples unique random number from 0 to len(background_inputs)
            random_indices = np.random.choice(len(background_inputs), max_samples, replace=False)

            # Add background samples to inputs
            for i in random_indices:
                self.inputs.append(background_inputs[i])

    def few_shot_inputs(self, num_samples):
        cls_dict = {}
        for item in self.inputs:
            _, _, _, target, _ = item
            if target not in cls_dict:
                cls_dict[target] = [item]
            else:
                cls_dict[target].append(item)
        
        select_vids = []
        for category, v in cls_dict.items():
            sample = random.sample(v, num_samples)
            select_vids.extend(sample)
        n_repeat = len(self.inputs) // len(select_vids)
        self.inputs = select_vids * n_repeat
        

    def __getitem__(self, index):
        video, start, end, target, future_targets = self.inputs[index]
        
        features = torch.load(os.path.join(self.data_path, f'{video}.pt'), weights_only=False)
        features = features[start:end]

        return features, target, future_targets

    def __len__(self):
        return len(self.inputs)