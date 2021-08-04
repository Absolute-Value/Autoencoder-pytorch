import torch, os
from torchvision import datasets, transforms

def get_loader(config='other', class_name='wrench', is_train=True, batch_size=32, img_nch=1, img_size=128):
    transform = get_transform(config, is_train, img_nch, img_size)
    dataset = get_dataset(config, class_name, is_train, transform, img_size)

    if is_train:
        zero_dataset = []
        if config == 'mnist' or config == 'fashion':
            for data_x, data_y in dataset:
                if data_y == int(class_name):
                    zero_dataset.append([data_x, data_y])
            dataset = zero_dataset
        n_samples = len(dataset)
        train_size = int(len(dataset) * 0.9)
        val_size = n_samples - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=is_train,
            drop_last=is_train,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )
        print('Train: {} ({}), Val: {} ({})'.format(train_size, len(train_loader), val_size, len(val_loader)))
        return train_loader, val_loader
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False,
            shuffle=is_train,
            drop_last=is_train,
        )
        print('Test: {} ({})'.format(len(dataset), len(loader)))
        return loader

def get_transform(config, is_train, img_nch, img_size):
    transform = []
    if config == 'mnist':
        transform = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    if config == 'fashion':
        transform = [
            transforms.Resize(img_size),
        ]
        if is_train is True:
            transform += [transforms.RandomHorizontalFlip(p=0.5)]
        transform += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    if config == 'mvtec':
        transform = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if img_nch==3 else transforms.Normalize([0.5], [0.5])
        ]
    if config == 'other':
        transform = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if img_nch==3 else transforms.Normalize([0.5], [0.5]) #最後がSigmoidのときはNormalize NG
        ]
    if config == 'cloth':
        transform = [transforms.Resize((img_size,img_size))]
        if img_nch == 1:
            transform.append(
                transforms.Grayscale(1)
            )
        if is_train is True:
            transform.append(
                transforms.RandomHorizontalFlip(p=0.5),
            )
        transform.append(
            transforms.ToTensor()
        )
        if img_nch == 3:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )
        else:
            transform.append(
                transforms.Normalize([0.5], [0.5])
            )
    return transforms.Compose(transform)

def get_dataset(config, class_name, is_train, transform, img_size):
    if config == 'mnist':
        dataset = datasets.MNIST(
            root='./data/MNIST/' + ('train' if is_train is True else 'test'),
            transform=transform,
            train=is_train,
            download=True
        )
    elif config == 'fashion':
        dataset = datasets.FashionMNIST(
            root='./data/FashionMNIST/' + ('train' if is_train is True else 'test'),
            transform=transform,
            train=is_train,
            download=True
        )
    elif config == 'mvtec':
        dataset = datasets.ImageFolder(
            root='./data/mvtec/{}/'.format(class_name) + ('train' if is_train is True else 'test'),
            transform=transform,
        )
    elif config == 'cloth':
        if img_size == 512:
            dataset = datasets.ImageFolder(
                root='../ClothRIAD/data/cloth512/{}/'.format(class_name) + ('train' if is_train is True else 'test'),
                transform=transform,
            )
        else:
            dataset = datasets.ImageFolder(
                root='../ClothRIAD/data/cloth256/{}/'.format(class_name) + ('train' if is_train is True else 'test'),
                transform=transform,
            )
    elif config == 'other':
        dataset = datasets.ImageFolder(
            root='./data/other/{}/'.format(class_name) + ('train' if is_train is True else 'test'),
            transform=transform,
        )
    else:
        raise ValueError('Invalid dataset: {}'.format(config.dataset))

    return dataset