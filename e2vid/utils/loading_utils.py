from e2vid.model.model import *
from collections import OrderedDict


def load_model(path_to_model, return_task=False):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model)
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    # E2VID Decoder
    decoder = E2VIDDecoder(model_type)
    decoder.load_state_dict(raw_model['state_dict'], strict=False)

    if return_task:
        # E2VID Task
        task = E2VIDTask(model_type)
        new_dict = copyStateDict(raw_model['state_dict'])
        keys = []
        for k, v in new_dict.items():
            if k.startswith('unetrecurrent.pred'):
                continue
            keys.append(k)

        new_dict = {k: new_dict[k] for k in keys}
        task.load_state_dict(new_dict, strict=False)
        return model, decoder, task
    return model, decoder


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])

        new_state_dict[name] = v
    return new_state_dict

