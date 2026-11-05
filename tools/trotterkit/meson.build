# SPDX-License-Identifier: GPL-3.0-or-later
project('trotterkit', 'c',
  version: '0.0.1',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['c_std=c99', 'buildtype=release',
                    'warning_level=3', 'werror=true'])

cc = meson.get_compiler('c')
m_dep = cc.find_library('m', required: false)
inc = include_directories('include')

shared_library('trotterkit_c',
  'backends/c/trotterkit_c.c',
  include_directories: inc,
  dependencies: m_dep,
  c_args: ['-march=native', '-ffp-contract=off'])
