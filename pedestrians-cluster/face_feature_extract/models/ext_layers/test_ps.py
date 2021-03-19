import torch
import torch.nn as nn
import numpy as np
from paramclient import ParameterClient


def test_sgd(bs, cls, fdim, midw, lr, weight_decay, momentum):

    assert cls >= bs
    assert lr > 0
    assert weight_decay >= 0
    assert momentum >= 0
    assert all([isinstance(x, int) for x in [bs, cls, fdim]])

    # setup model
    logits = nn.Linear(fdim, cls, bias=False)
    params = [logits.weight]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params,
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=momentum)

    # setup ps client
    client = ParameterClient(0)
    client.add_matrix(midw, [cls, fdim])
    client.update_params({
        'lr': lr,
        'weight_decay': weight_decay,
        'momentum': momentum
    })
    client.set_matrix(midw, logits.weight.data.numpy(), force=True)
    rows = np.arange(cls)

    for i in range(10):
        ps_weight = client.get_value_by_rows(midw, rows)
        th_weight = logits.weight.data.detach_().numpy()
        # updated weight computed by parameter server
        # should be equal to the result of PyTorch
        assert np.allclose(ps_weight, th_weight), \
                "{}-th: ps vs torch({} vs {})".format(i, ps_weight.mean(), th_weight.mean())

        # one-layer training
        x = torch.randn(bs, fdim)
        target = torch.LongTensor(range(bs))
        y = logits(x)
        loss = criterion(y, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # parameter server sgd update
        # skip_decay=True since the grad has already been regularized
        client.update_by_rows(midw,
                              rows,
                              logits.weight.grad.numpy(),
                              skip_decay=True)


if __name__ == '__main__':
    """ This testing script is to ensure that
            1. There is no precision loss during the transportation between client and server.
            2. The sgd algorithm of parameter server is compatible with PyTorch.
    """
    test_cases = [{
        'bs': 2,
        'cls': 3,
        'fdim': 4,
        'midw': '0',
        'lr': 0.01,
        'weight_decay': 0,
        'momentum': 0.9
    }, {
        'bs': 2,
        'cls': 3,
        'fdim': 4,
        'midw': '0',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0
    }, {
        'bs': 32,
        'cls': 1000,
        'fdim': 256,
        'midw': '1',
        'lr': 0.01,
        'weight_decay': 0,
        'momentum': 0.9
    }, {
        'bs': 32,
        'cls': 1000,
        'fdim': 256,
        'midw': '1',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0.0
    }, {
        'bs': 32,
        'cls': 1000,
        'fdim': 256,
        'midw': '1',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'momentum': 0.9
    }]
    for i, t in enumerate(test_cases):
        try:
            test_sgd(t['bs'], t['cls'], t['fdim'], t['midw'], t['lr'],
                     t['weight_decay'], t['momentum'])
            print('[Passed] {}-th test case: {}'.format(i, t))
        except Exception as err:
            print('[Failed] {}-th test case: {} ({})'.format(i, t, err))
