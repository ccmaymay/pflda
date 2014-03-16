# vim: ft=python

BUILD_DIR = '#build'

env = Environment()
env.Append(BUILD_DIR=BUILD_DIR)

Export('env')

env.SConscript('src/SConscript', variant_dir=BUILD_DIR) #, duplicate=0)
