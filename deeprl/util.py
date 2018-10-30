#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import inspect
import logging
import zipfile
import platform

import numpy as np

from unityagents import UnityEnvironment

try:
    from IPython.display import Markdown, display
except Exception:
    logging.exception('Failed to import jupyter modules. Ignoring')

here = os.path.dirname(os.path.abspath(__file__))

FRAME_SKIP = 1


class UnityEnvironmentWrapper(object):
    def __init__(self, env, frameskip=FRAME_SKIP):
        self.env = env
        self.frameskip = frameskip

    def step(self, action):
        for i in range(self.frameskip):
            env = self.env.step(action)
        return env

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def rgb2gray(img):
    img = img.squeeze()
    if len(img.shape) == 3:
        return img.dot(RGB_TO_GRAY_WEIGHTS).reshape(
            1, img.shape[0], img.shape[1], 1
        )
        #return (img ** 2).dot(
        #    [0.299, 0.587, 0.114]
        #).reshape(3, img.shape[0], img.shape[1], 1)
    else:
        raise ValueError('Image in some color space not known')


def get_state(env_info, use_visual=False, n_agents=1):
    if use_visual:
        state = env_info.visual_observations[0]
        state = rgb2gray(state).transpose(0, 3, 1, 2)
    else:
        if n_agents > 1:
            state = env_info.vector_observations
        else:
            state = env_info.vector_observations[0]
    return state


def build_environment(path):
    macos = path.endswith('.app')
    if macos:
        directory = path
    else:
        directory = os.path.dirname(path)
    if not os.path.exists(directory):
        zipped = directory + '.zip'
        if os.path.exists(zipped):
            with zipfile.PyZipFile(zipped) as zfp:
                logging.info('Extracting environment...')
                zfp.extractall()
        else:
            raise Exception('Unable to proceed. Cannot find environment.')
    if macos:
        for root, _, files in os.walk(path):
            for fn in files:
                os.chmod(os.path.join(root, fn), 0o755)
    os.chmod(path, 0o755)
    return UnityEnvironment(file_name=path)


def get_executable_path():
    parent = os.path.dirname(here)
    if platform.system() == 'Linux':
        if '64' in platform.architecture()[0]:
            return os.path.join(parent, 'Tennis_Linux', 'Tennis.x86_64')
        else:
            return os.path.join(parent, 'Tennis_Linux', 'Tennis.x86')
    elif platform.system() == 'Darwin':
            return os.path.join(parent, 'Tennis.app')
    elif platform.system() == 'Windows':
        if '64' in platform.architecture()[0]:
            return os.path.join(parent, 'Tennis_Windows_x86_64', 'Tennis.exe')
        else:
            return os.path.join(parent, 'Tennis_Windows_x86', 'Tennis.exe')
    else:
        logging.error('Unsupported platform!')
        raise ValueError('Unsupported platform')


def print_source(obj):
    source = inspect.getsource(obj)
    display(Markdown('```python\n' + source + '\n```'))


def load_environment():
    path = get_executable_path()
    return build_environment(path)
