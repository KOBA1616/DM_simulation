Headless testing helpers

Use `dm_toolkit.gui.headless` to create and drive `GameSession` instances without
importing any Qt modules. This is intended for unit tests or CI where a GUI
isn't available.

Examples

Python:

```py
from dm_toolkit.gui.headless import create_session, play_instance, run_steps

sess = create_session(p0_human=True)
# find a playable instance id from logs or via sess.gs.players
# Suppose instance 123 is in hand and legal
ok = play_instance(sess, 123)
steps, over = run_steps(sess, max_steps=200)
```

Notes

- `create_session` will try to load `data/cards.json` via the same robust loader
  used by the GUI when `card_db` is not provided.
- These helpers avoid PyQt imports so tests can run in headless environments.
