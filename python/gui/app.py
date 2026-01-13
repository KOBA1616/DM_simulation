#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_dir = os.path.join(root_dir, 'bin')

if root_dir not in sys.path:
    sys.path.append(root_dir)
if bin_dir not in sys.path:
    sys.path.append(bin_dir)


from dm_toolkit.gui.app import main

if __name__ == "__main__":
    main()
