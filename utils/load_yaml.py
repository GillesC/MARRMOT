import os

import yaml
from os.path import dirname, abspath, join as pjoin

def load_root_dir():
    import yaml
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config.yml")) as file:
        return yaml.full_load(file)["root_dir"]



def get_conf(force_reload=False):
    root_dir = dirname(__file__)
    conf_path = abspath(pjoin(root_dir, "..", "config.yml"))
    conf = None
    with open(conf_path, 'r') as stream:
        try:
            conf = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return conf

def get_data_path():
    conf = load_conf()


    # class Config:
    # root_dir = dirname(__file__)
    # conf_path = abspath(pjoin(root_dir, "..", "config.yml"))

    # def __init__(self):
    #     print(f'Loading configuration at {conf_path}')
    #     with open(conf_path, 'r') as stream:
    #         try:
    #             self.config = yaml.safe_load(stream)
    #         except yaml.YAMLError as exc:
    #             print(exc)

    # def __getattr__(self, name):
    #     try:
    #         return self.config[name]
    #     except KeyError:
    #         return getattr(self.args, name)

