# -*- coding: utf-8 -*-

import os
import json

LOG_DIR = os.path.join(os.path.dirname(os.path.realpath("__file__")), "log")


def create_log_dict(args):

    log_dict = {}

    for key, value in vars(args).items():
        log_dict[key] = value

    return log_dict


def save_log(log_dict, log_filename):

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    with open(os.path.join(LOG_DIR, log_filename), "w") as wf:
        json.dump(log_dict, wf, indent=2)
