#!/bin/bash

# commit that introduced --cli flag to buildlocal
COMPSS_CLI_COMMIT=b23b5c8dda337a09021e34232d49227d1b67c22e

COMPSS_TAG=${1}
COMPSS_BINDING_C=${2}
COMPSS_BINDING_PYTHON=${3}
COMPSS_EXTRAE=${4}
COMPSS_AUTOPARALLEL=${5}
COMPSS_SKIP_TESTS=${6}
COMPSS_MONITOR=${7}
COMPSS_STREAM=${8}
COMPSS_JACOCO=${9}
COMPSS_KAFKA=${10}
COMPSS_CLI=${11}

git clone https://github.com/bsc-wdc/compss.git /opt/compss-repo
cd /opt/compss-repo
git checkout ${COMPSS_TAG}
./submodules_get.sh
./submodules_patch.sh

args=()
if [ "$COMPSS_BINDING_C" = "true" ]; then \
	args+=(--bindings)
fi
if [ "$COMPSS_BINDING_PYTHON" = "true" ]; then \
	args+=(--pycompss)
fi
if [ "$COMPSS_EXTRAE" = "true" ]; then \
	args+=(--tracing)
fi
if [ "$COMPSS_AUTOPARALLEL" = "true" ]; then \
	args+=(--autoparallel)
fi
if [ "$COMPSS_MONITOR" = "true" ]; then \
	args+=(--monitor)
fi
if [ "$COMPSS_STREAM" = "true" ]; then \
	args+=(--stream)
fi
if [ "$COMPSS_JACOCO" = "true" ]; then \
	args+=(--jacoco)
fi
if [ "$COMPSS_KAFKA" = "true" ]; then \
	args+=(--kafka)
fi
if [ "$COMPSS_CLI" = "true" ] && [ $(git merge-base --is-ancestor $COMPSS_CLI_COMMIT $COMPSS_TAG) ]; then \
	args+=(--cli)
fi
if [ "$COMPSS_SKIP_TESTS" = "true" ]; then \
	args+=(--skip-tests)
fi

cd builders
./buildlocal --skip-tests --nothing ${args[@]} /opt/COMPSs

rm -r /opt/compss-repo