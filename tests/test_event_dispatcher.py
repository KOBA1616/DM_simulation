import threading
import time
from dm_toolkit.event_dispatcher import EventDispatcher


def test_subscribe_emit_unsubscribe():
    ed = EventDispatcher()
    results = []

    def cb(ev):
        results.append(ev)

    token = ed.subscribe('TST', cb)
    ed.emit('TST', {'v': 1})
    assert results == [{'v': 1}]

    removed = ed.unsubscribe(token)
    assert removed
    ed.emit('TST', {'v': 2})
    assert results == [{'v': 1}]


def test_threaded_emit():
    ed = EventDispatcher()
    acc = []

    def cb(ev):
        acc.append(ev['i'])

    ed.subscribe('THREAD', cb)

    def worker(start, end):
        for i in range(start, end):
            ed.emit('THREAD', {'i': i})

    t1 = threading.Thread(target=worker, args=(0, 50))
    t2 = threading.Thread(target=worker, args=(50, 100))
    t1.start(); t2.start(); t1.join(); t2.join()

    # ensure 100 events handled
    assert sorted(acc) == list(range(100))
