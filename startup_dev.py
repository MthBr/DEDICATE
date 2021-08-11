#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I have enabled a file to be run whenever I run a file.
That file consists of a script to clear all the variables.
https://github.com/spyder-ide/spyder/issues/2563

"""
def init(): 
    #Clean Memory and import pakage with folders
    for i in list(globals().keys()):
        if(i[0] not in ['_', 'load_test_file','watershed_segmentation' ]):
            exec('del {}'.format(i))


def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]


if __name__ == "__main__":
    clear_all()
    init()