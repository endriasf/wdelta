#!/usr/bin/env bash
# Build the release files for one platform, then test them as a user would get them.
#
#   build_release.sh TAG ASSET WHEEL_PLATFORM_TAG PYTHON
#
# Produces in dist/:
#   wdelta-TAG-ASSET.tar.gz (or .zip on Windows)   the command-line codec
#   wdelta-VERSION-py3-none-PLATFORM.whl           the Python package with the codec bundled
set -euo pipefail
tag=$1 asset=$2 plat=$3 py=$4
version=${tag#v}

# the tag must match the version in both package files
grep -q "^version = \"$version\"" Cargo.toml || { echo "Cargo.toml version is not $version"; exit 1; }
grep -q "^version = \"$version\"" python/pyproject.toml || { echo "python/pyproject.toml version is not $version"; exit 1; }

cargo build --release --locked
exe=target/release/wdelta
[ -f "$exe.exe" ] && exe="$exe.exe"
"$py" tests/golden_check.py "$exe"

mkdir -p dist
# command-line archive
name="wdelta-$tag-$asset"
mkdir "$name"
cp "$exe" LICENSE NOTICE README.md "$name/"
if [[ "$exe" == *.exe ]]; then
  7z a "dist/$name.zip" "$name" > /dev/null
else
  tar czf "dist/$name.tar.gz" "$name"
fi

# Python wheel with the codec bundled, tagged for this platform only
mkdir -p python/wdelta/bin
cp "$exe" python/wdelta/bin/
"$py" -m pip install --quiet --upgrade pip build wheel
"$py" -m build --wheel --outdir wheelhouse python
"$py" -m wheel tags --python-tag py3 --abi-tag none --platform-tag "$plat" --remove wheelhouse/*.whl
cp wheelhouse/*.whl dist/

# test the wheel the way a user installs it: fresh environment, no source tree, no WDELTA_BIN
venv="$RUNNER_TEMP/wheel-test"
"$py" -m venv "$venv"
vpy="$venv/bin/python"; [ -f "$venv/Scripts/python.exe" ] && vpy="$venv/Scripts/python.exe"
"$vpy" -m pip install --quiet dist/*.whl numpy pytest
mkdir -p "$RUNNER_TEMP/wheel-tests"
cp python/tests/test_package.py "$RUNNER_TEMP/wheel-tests/"
(
  cd "$RUNNER_TEMP/wheel-tests"
  unset WDELTA_BIN
  "$vpy" -c "
import wdelta, os
b = wdelta._binary()
print('installed', wdelta.__version__, 'from', wdelta.__file__, '| binary', b)
assert wdelta.__version__ == '$version', wdelta.__version__
assert 'site-packages' in wdelta.__file__ and 'site-packages' in b
"
  "$vpy" -m pytest -q test_package.py
)
ls -l dist
