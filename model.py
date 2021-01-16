import segmentation_models_pytorch as smp
import torch

def init_model(
        path='saved_models/unet_model.pth',
        from_scratch=True,
        best=False,
        device=torch.device('cpu')):
    """
    We are using the models from the segmentation models pytorch library.
    This is a wrapper around it to help us with creation and loading
    """
    model = smp.UnetPlusPlus(encoder_name='resnet34',
                   encoder_weights='imagenet',
                   in_channels=1,
                   classes=1).to(device)
    if from_scratch:
        print('Model initialised from Scratch.')
        return model
    elif best:
        path = path.split('.')
        path[0] += '_best'
        path = '.'.join(path)
        model.load_state_dict(torch.load(path))
        print('Loaded saved model at: ', path)
        return model
    else:
        model.load_state_dict(torch.load(path))
        print('Loaded saved model at: ', path)
        return model
