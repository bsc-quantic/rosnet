#!/bin/bash

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
if [ "$COMPSS_CLI" = "true" ]; then \
	args+=(--cli)
fi
if [ "$COMPSS_SKIP_TESTS" = "true" ]; then \
	args+=(--skip-tests)
fi

cd builders
./buildlocal --skip-tests --nothing ${args[@]} /opt/COMPSs

rm -r /opt/compss-repo