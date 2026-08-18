project(
  'paulikit',
  'c', 'cpp', 'cython',
  version: '0.1.0',
  license: 'GPL-3.0-or-later',
  meson_version: '>=1.1.0',
  default_options: ['cpp_std=c++17'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)

subdir('src/paulikit')
