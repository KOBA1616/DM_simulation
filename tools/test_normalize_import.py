from dm_toolkit.gui.editor import normalize
print('HAS', hasattr(normalize,'canonicalize'))
print(normalize.canonicalize({'type':'TEST'}))
