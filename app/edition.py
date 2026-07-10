"""Edition flag: admin (full) vs user (writers') build.

The user edition is produced by deploy-to-user.bat, which copies the
sources WITHOUT the admin-only modules (AI writers, claude bridge, VO,
translate, upscale, FCPXML, render) and drops a USER_EDITION marker file
at the project root. Everything edition-specific in the codebase keys off
IS_USER — there are no forked copies of main_window.py / script_panel.py
to maintain (the old fork approach lost the user files once already).
"""

import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IS_USER = os.path.exists(os.path.join(_ROOT, "USER_EDITION"))
