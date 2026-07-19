#!/usr/bin/env bash
# Builds the vendored ITURHFProp into baselines/p533/bin/iturhfprop.
# Portable: macOS (clang) and Linux (gcc/clang). No network access.
set -euo pipefail
cd "$(dirname "$0")"

UP=upstream
OUT=bin
mkdir -p "$OUT"

CC="${CC:-cc}"
# -fcommon: several ITURHFProp/Src/ITURHFProp/*.c files declare the same
# dll*/hLib function-pointer globals (for runtime dlsym resolution of the
# noise library) without `extern`, so each translation unit tentatively
# defines them. GCC 10+ defaults to -fno-common on Linux and fails those
# as "multiple definition" link errors; clang (macOS's default cc) doesn't
# hit this the same way, so it only surfaces on Linux. -fcommon restores
# the traditional tentative-definition merging this legacy C relies on.
CFLAGS="-O2 -fPIC -fcommon -w"

# Reality check against the vendored tree's own makefiles (P372/Linux/Makefile,
# P533/Linux/Makefile, ITURHFProp/Linux/Makefile), confirmed by test-building
# with them directly before writing this script:
#
# 1. Source lives one level deeper than the top-level convention would
#    suggest: P372/Src/P372/*.c, P533/Src/P533/*.c, ITURHFProp/Src/ITURHFProp/*.c
#    (not P372/Src/*.c etc).
# 2. P372/Src/P372/ also contains NoiseDriver.c, which defines its own main()
#    and is NOT part of the noise library (upstream's own Makefile excludes
#    it too) -- list libp372's sources explicitly rather than globbing, or
#    the build still succeeds (multiple mains are legal in a shared object)
#    but pulls in an unused, unbuilt-for-this-purpose entry point.
# 3. ITURHFProp does NOT link against libp533/libp372 at compile time. It
#    dlopen()s them by bare filename ("libp533.so", "libp372.so") at runtime
#    (ITURHFProp.c, ~lines 119 and 329) via dlsym-resolved function pointers.
#    So the third compile line below only needs -lm -ldl, matching upstream's
#    own ITURHFProp/Linux/Makefile exactly -- and the *caller* of the binary
#    must arrange for libp533.so/libp372.so to be found by the dynamic
#    linker's dlopen() search: set LD_LIBRARY_PATH (Linux) or
#    DYLD_LIBRARY_PATH (macOS) to this bin/ directory before running
#    iturhfprop. (Task 4's subprocess wrapper handles this via env=.)

$CC $CFLAGS -shared -o "$OUT/libp372.so" \
    "$UP/P372/Src/P372/InitializeNoise.c" \
    "$UP/P372/Src/P372/Noise.c" \
    "$UP/P372/Src/P372/NoiseMemory.c" \
    -lm -ldl

$CC $CFLAGS -shared -o "$OUT/libp533.so" $UP/P533/Src/P533/*.c -lm -ldl

$CC $CFLAGS -o "$OUT/iturhfprop" $UP/ITURHFProp/Src/ITURHFProp/*.c -lm -ldl

echo "built: $OUT/iturhfprop"
echo "note: set LD_LIBRARY_PATH/DYLD_LIBRARY_PATH=$(cd "$OUT" && pwd) before running it directly"
