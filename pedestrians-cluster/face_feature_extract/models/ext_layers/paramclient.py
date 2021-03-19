import zmq
import json
import numpy as np
import time


class ParameterClient():
    def __init__(self, _id):
        print('Connecting to ParamServer ...')
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = u'%d' % _id
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:5570')
        print('Client %s started' % (identity))
        self._poll = zmq.Poller()
        self._poll.register(socket, zmq.POLLIN)
        self._socket = socket
        self._context = context
        self._register()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Client %s exit' % (self._socket.identity))
        self._exit()
        self._socket.close()
        self._context.term()

    def _register(self):
        msg = dict(op='register')
        self._socket.send_json(msg)

    def _exit(self):
        msg = dict(op='exit')
        self._socket.send_json(msg)

    def _send_array(self, data, flags=0, copy=True, track=False):
        """send a numpy array with metadata"""
        md = dict(
            dtype=str(data.dtype),
            shape=data.shape,
        )
        self._socket.send_json(md, flags | zmq.SNDMORE)
        return self._socket.send(data, flags, copy=copy, track=track)

    def add_matrix(self, mid, shape):
        msg = dict(op='add_matrix', mid=mid, shape=shape)
        self._socket.send_json(msg)

    def set_matrix(self, mid, data, force=False):
        msg = dict(op='set_matrix', mid=mid, force=force)
        self._socket.send_json(msg, zmq.SNDMORE)
        self._send_array(data)

    def get_value_by_rows(self, mid, rows):
        msg = dict(op='get_value_by_rows', mid=mid)
        self._socket.send_json(msg, zmq.SNDMORE)
        self._send_array(rows)
        # receive data
        meta = None
        while True:
            sockets = dict(self._poll.poll(1000))
            if self._socket in sockets:
                msg = self._socket.recv()
                if not meta:
                    meta = json.loads(msg)
                else:
                    data = np.frombuffer(msg, dtype=meta['dtype'])
                    return data.reshape(meta['shape'])

    def set_value_by_rows(self, mid, rows, data):
        msg = dict(
            op='set_value_by_rows',
            mid=mid,
        )
        self._socket.send_json(msg, zmq.SNDMORE)
        self._send_array(rows, zmq.SNDMORE)
        self._send_array(data)

    def update_params(self, dic):
        assert len(dic) > 0
        msg = dict(op='update_params')
        msg.update(dic)
        self._socket.send_json(msg)

    def update_by_rows(self, mid, rows, data, skip_decay=False):
        msg = dict(op='update_by_rows', mid=mid, skip_decay=skip_decay)
        self._socket.send_json(msg, zmq.SNDMORE)
        self._send_array(rows, zmq.SNDMORE)
        self._send_array(data)

    def snapshot(self, path):
        msg = dict(op='snapshot', path=path)
        self._socket.send_json(msg)

    def load(self, path):
        msg = dict(op='resume', path=path)
        self._socket.send_json(msg)

    def resume(self, path):
        msg = dict(op='resume', path=path)
        self._socket.send_json(msg)


if __name__ == '__main__':
    num_class, fdim = 10, 256
    client0 = ParameterClient(0)
    client1 = ParameterClient(1)
    # make sure all the clients have successfully setup
    import time
    time.sleep(3)
    client0.add_matrix(mid='0', shape=[num_class, fdim])
    weights = client0.get_value_by_rows(mid='0', rows=[0, 1, 2, 3])
    client0.update_by_rows(mid='0', rows=[0, 1, 2, 3], data=np.ones([4, fdim]))
    client1.update_by_rows(mid='0', rows=[2, 3, 4, 5], data=np.ones([4, fdim]))
