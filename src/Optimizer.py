"""

"""

import torch.optim as optim


def get_optimizer(parameters, settings):
    """
    read the input file, and create the optimizer object that is used to update the networks weights and biases
    :param parameters: the weights and biases of the neural network
    :param settings: a dictionary containing the data hyper parameters for the optimizer
    :return: the built optimizer object
    """

    if settings['name'] == 'ADAM':
        # create an ADAM based optimzier
        optimizer = optim.Adam(parameters, lr=settings['learn_rate'], betas=(settings['beta_1'], settings['beta_2']))
    elif settings['name'] == 'RMS':
        optimizer = optim.RMSprop(parameters, lr=settings['learn_rate'],alpha=settings['alpha'],
                                  momentum=settings['momentum'])
    else:
        raise ValueError('Un supported optimizer. Either add an optimizer or switch to a supported one')

    return optimizer