# The compiled fast path for paulikit.algorithms.fwht.fwht_pauli_terms.
# Optional (see meson.options's "native" feature) so paulikit stays
# pip-installable without a C++ toolchain or oneTBB; fwht.py falls
# back to pure Python with a visible warning if this module isn't
# built. See PLAN.md's packaging note - this optional/fallback model
# is a deliberate temporary compromise, not the eventual target (the
# NumPy/SciPy model - extension always required, shipped as prebuilt
# wheels - is the goal once wheel-building CI exists).

native_opt = get_option('native')

if not native_opt.disabled()
  cy = meson.get_compiler('cython')
  tbb_dep = dependency('tbb', required: native_opt)
  cython_ok = cy.version().version_compare('>=3.0')

  if tbb_dep.found() and cython_ok
    np_include_path = run_command(
      py, ['-c', 'import numpy; print(numpy.get_include())'],
      check: true,
    ).stdout().strip()

    py.extension_module(
      'pauli_label_native',
      [
        'pauli_label_native.pyx',
        'pauli_label.c',
        'pauli_label_parallel.cpp',
      ],
      include_directories: include_directories('.', np_include_path, is_system: true),
      dependencies: [tbb_dep],
      override_options: ['cython_language=cpp'],
      install: true,
      subdir: 'paulikit/_native',
    )
  elif native_opt.enabled()
    error('native extension explicitly requested (-Dnative=enabled) but ' +
          'oneTBB (found: @0@) or Cython >= 3.0 (found: @1@) is missing'.format(
            tbb_dep.found(), cython_ok))
  else
    message('paulikit: native extension not built (oneTBB or Cython >= 3.0 ' +
            'not found) - falling back to pure Python at import time')
  endif
else
  message('paulikit: native extension disabled via -Dnative=disabled')
endif

# The empirical cache-latency probe (PLAN.md Phase 12) - standalone,
# Cython-only, deliberately independent of the native/oneTBB option
# above (the probe has no relationship to oneTBB; see cache_probe.pyx's
# own docstring for why bundling them would be wrong).
cache_probe_opt = get_option('cache_probe')

if not cache_probe_opt.disabled()
  cy2 = meson.get_compiler('cython')
  cython_ok2 = cy2.version().version_compare('>=3.0')

  if cython_ok2
    py.extension_module(
      'cache_probe',
      [
        'cache_probe.pyx',
        'cache_probe.c',
      ],
      include_directories: include_directories('.'),
      install: true,
      subdir: 'paulikit/_native',
    )
  elif cache_probe_opt.enabled()
    error('cache_probe extension explicitly requested (-Dcache_probe=enabled) but ' +
          'Cython >= 3.0 was not found')
  else
    message('paulikit: cache_probe extension not built (Cython >= 3.0 not found) - ' +
            'falling back to declared-size cache detection at runtime')
  endif
else
  message('paulikit: cache_probe extension disabled via -Dcache_probe=disabled')
endif

# The __init__.py for the _native package itself (plain Python,
# always installed regardless of whether the extension built).
py.install_sources(
  '__init__.py',
  subdir: 'paulikit/_native',
)
