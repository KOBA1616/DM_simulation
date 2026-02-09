import os
import sys
os.environ['DM_DISABLE_NATIVE'] = '1'
import pytest
if __name__ == '__main__':
    sys.exit(pytest.main(['-q']))
