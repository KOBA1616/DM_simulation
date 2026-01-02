#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os

# Add project root to path if not present
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from dm_toolkit.gui.card_editor import main

if __name__ == "__main__":
    main()
