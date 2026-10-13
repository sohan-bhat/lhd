import numpy as np
from tiny_net import TinyAdapter, TinyNet

from halt.baseline import load_history
from halt.recorder import Recorder


def test_load_history_reads_saved_run(tmp_path):
    net = TinyNet()
    net.dW1 = np.ones((3, 4))
    adapter = TinyAdapter(net)
    recorder = Recorder(adapter, probe_batch=np.ones((5, 3)))
    recorder.start_epoch()
    recorder.after_batch()
    recorder.end_epoch(epoch=1, train_accuracy=0.5, val_accuracy=0.4, seconds=1.0)
    path = tmp_path / "run.json"
    recorder.save(str(path), metadata={"seed": 0})
    assert len(load_history(str(path))) == 1
