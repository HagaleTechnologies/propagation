# ITURHFProp provenance

- Upstream: https://github.com/ITU-R-Study-Group-3/ITU-R-HF.git
- Version: commit `82017594a1c6cacfaa7e86954c4ae7b3a5825a3d`, dated 2025-05-15T10:39:37-06:00
- Fetched: 2026-07-17 by Claude Code (agent session, on behalf of Tony Hagale / K5ARH)
- Tree sha256: `9ab02d92502ea085631d6e6031d358c619c271de40c38773e688bdd9e82c2b92`
  (computed with GNU tar; see command below — macOS's bundled BSD tar does not
  support `--sort`, install `gnu-tar` via Homebrew and use `gtar` if
  reproducing this on macOS)
- License: **no separate LICENSE file exists in the upstream tree** (confirmed
  via `find` for LICENSE/COPYING/TERMS variants, and via the GitHub API, which
  reports `license: null` for this repo). The terms are stated inline at the
  bottom of `upstream/README.md` (lines 1098-1113): the software "may be used
  by implementers in their implementation of the Recommendation as well as in
  revisions of the specific original Recommendation and in other ITU
  Recommendations, free from any copyright assertions," provided "as is" with
  no warranties, and the ITU disclaims liability. This is **not** an
  OSI-approved open-source license and is **not** MIT/Apache-2.0 — it is a
  narrower, ITU-specific grant for implementers of the Recommendation. This
  vendored subtree (`baselines/p533/upstream/`) is therefore carved out from
  this repo's own MIT OR Apache-2.0 dual license; it retains the ITU's terms
  as stated in `upstream/README.md`.
- Local modifications:
  - `build.sh` drives compilation directly because upstream's own
    `*/Linux/Makefile`s are hardcoded to build in-place under each
    component's `Linux/` directory and don't compose cleanly across a
    vendored, relocated tree. `build.sh`'s three compile lines were adapted
    from (and verified against) the real upstream Makefiles:
    `P372/Linux/Makefile`, `P533/Linux/Makefile`,
    `ITURHFProp/Linux/Makefile`. Notably: source lives one level deeper than
    the top-level convention would suggest (`P372/Src/P372/*.c`, not
    `P372/Src/*.c`); `ITURHFProp` does not link `libp533`/`libp372` at
    compile time, it `dlopen()`s them by bare filename at runtime — see
    `build.sh` for the full note and the runtime env var requirement this
    implies for Task 4's subprocess wrapper.
  - No patches to any vendored `.c`/`.h` file. Any future patch must be
    listed here with a rationale and kept in `baselines/p533/patches/`.
  - The vendored tree is a **trimmed subset** of the upstream clone, not a
    byte-for-byte mirror (the tree sha256 above is over this trimmed subset,
    not the full upstream clone). Excluded, and why:
    - `.git/`, `.github/` — VCS/CI metadata, not source.
    - `*/Win32/` (all three components) — Visual Studio project files for a
      platform this repo doesn't build on.
    - `*/Bin/` (all three components) — prebuilt Windows/Linux binaries,
      debug symbols (`.pdb`), and sample `.in`/`.out` test-run artifacts.
    - `*/Linux/*.so` and `ITURHFProp/Linux/ITURHFProp` — prebuilt Linux
      binaries checked into the upstream repo. These were confirmed
      **stale**: they `dlopen`-error-string-match an older source revision
      that doesn't match what's actually in `Src/` today (verified via
      `strings`/`nm` on the prebuilt binary vs. `grep` on current source).
      `build.sh` rebuilds these from source instead of trusting them.
    - `ITURHFProp/Src/P533/` — not source at all: a stray, upstream-committed
      build-artifact directory containing only `.o`/`.d` object files and a
      VS Code `.vscode/ipch` IntelliSense cache (14 MB), no `.c`/`.h` files.
    - `P372/Src/AtmosPlots/ITURNoise.exe`, `P372/Src/AtmosPlots/P372.dll` —
      prebuilt Windows binaries embedded in an otherwise-source directory
      (the Python scripts there were kept).
    - `P533/Data/`, `P372/Data/` — confirmed byte-identical duplicates of
      `ITURHFProp/Data/` via `md5` spot-check (each component ships its own
      copy of the same coefficient dataset for standalone testing); keeping
      one copy avoids ~264 MB of redundant binary data.
    - `ITURHFProp/Data/Antenna/` (1,065 files, ~139 MB) — antenna radiation
      pattern library. Not required for this baseline: all path-model runs
      here use `TXAntFilePath "ISOTROPIC"` / `RXAntFilePath "ISOTROPIC"`
      (confirmed against upstream's own sample test cases), which needs no
      antenna pattern file.
- Coefficient data: `upstream/ITURHFProp/Data/` (ionospheric coefficient
  files — `COEFF*.BIN`/`.txt`, `ionos*.bin`, `FOF2*`, `P1239-3 Decile
  Factors.txt`) — required at runtime via the input file's `DataFilePath`
  parameter; shipped as part of this vendored tree. Verified end-to-end: a
  hand-built `.in` file pointed at this directory produced a correct P.533
  report (path geometry, MUF, noise, reliability) when run through the
  binary built by `build.sh`.
- cqdx cross-reference: sidecar vendor commit unverified — no cqdx access was
  used or consulted for this task (per repo policy: no cqdx code imports,
  ever). Agreement spot-check lives in scripts/p533_crosscheck.py (not CI,
  not part of this task).

Tree hash command:
    tar --sort=name --mtime='1970-01-01' -cf - baselines/p533/upstream | sha256sum
(on macOS: `brew install gnu-tar` and substitute `gtar` for `tar`, `shasum -a 256` for `sha256sum`)
