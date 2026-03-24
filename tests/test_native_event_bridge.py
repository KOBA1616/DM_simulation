from dm_toolkit import native_event_bridge, event_types


def test_native_emit_forwards_to_subscribers():
    results = []

    def cb(ev):
        results.append(ev)

    token = native_event_bridge.subscribe(event_types.STATE_CHANGED, cb)
    native_event_bridge.native_emit(event_types.STATE_CHANGED, {'s': 'ok'})
    assert results == [{'s': 'ok'}]
    assert native_event_bridge.unsubscribe(token)
