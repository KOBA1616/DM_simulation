import sys
sys.path.insert(0, r'C:\Users\ichirou\DM_simulation')
try:
    import dm_ai_module as dm
except Exception as e:
    print('IMPORT FAILED:', type(e).__name__, e)
else:
    print('IS_NATIVE=', getattr(dm,'IS_NATIVE', None))
    print('native module loaded=', getattr(dm,'__native_module__', None) is not None)
    # print sample attributes count
    attrs = [x for x in dir(dm) if not x.startswith('_')]
    print('SAMPLE COUNT=', len(attrs))
