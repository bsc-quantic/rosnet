ARG PYTHON_VERSION=3.9
FROM python:${PYTHON_VERSION}-bullseye

ARG DEBIAN_FRONTEND=noninteractive

# install system dependencies
RUN apt update; apt upgrade -y; \
	apt install -y --no-install-recommends git vim wget jq curl openssh-server sudo software-properties-common unzip; \
	# add-apt-repository universe; \
	git config --global core.compression 9; \
	apt install -y --no-install-recommends \
		maven \
		# COMPSs runtime
		openjdk-11-jdk graphviz xdg-utils \
		# COMPSs bindings-common
		libtool automake build-essential \
		# COMPSs misc
		openmpi-bin libopenmpi-dev uuid-runtime bc; \
	apt-get autoclean;

ARG COMPSS_BINDING_C=true
ARG COMPSS_BINDING_PYTHON=true
ARG COMPSS_EXTRAE=false
ARG COMPSS_AUTOPARALLEL=false

# optional features
# COMPSs C-binding
RUN if [ "$COMPSS_BINDING_C" = "true" ]; then \
		apt install -y --no-install-recommends libboost-all-dev libxml2-dev; \
	fi;

# COMPSs Python-binding
RUN	if [ "$COMPSS_BINDING_PYTHON" = "true" ]; then \
		pip install --upgrade pip; \
		pip install --no-cache-dir wheel dill guppy3 decorator mpi4py \
		# Pycodestyle
		pycodestyle \
		# Numba
		llvmlite>=0.31.0 numba \
		# Testing
		nose matplotlib pytest pytest-cov coverage; \
		# pytest-notebook nbval \
	fi;

# COMPSS Extrae
RUN	if [ "$COMPSS_EXTRAE" = "true" ]; then \
		apt-install -y --no-install-recommends libxml2 gfortran libpapi-dev papi-tools; \
	fi;

# COMPSs AutoParallel
RUN	if [ "$COMPSS_AUTOPARALLEL" = "true" ]; then \
		apt install -y --no-install-recommends libgmp3-dev flex bison libbison-dev texinfo libffi-dev; \
		if [ "$COMPSS_BINDING_PYTHON" = "true" ]; then \
			pip install --no-cache-dir astor sympy enum34 islpy; \
		fi; \
	fi;

# config ssh
RUN yes yes | ssh-keygen -f /root/.ssh/id_rsa -t rsa -N '' > /dev/null; \
	cat /root/.ssh/id_rsa.pub > /root/.ssh/authorized_keys; \
	mkdir /run/sshd

# configure user environment
ENV GRADLE_HOME=/opt/gradle
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
ENV MPI_HOME=/usr/lib/openmpi
ENV COMPSS_HOME=/opt/COMPSs/
ENV PATH=$PATH:/opt/COMPSs/Runtime/scripts/user:/opt/COMPSs/Bindings/c/bin:/opt/COMPSs/Runtime/scripts/utils:/opt/gradle/bin
ENV LD_LIBRARY_PATH=/opt/COMPSs/Bindings/bindings-common/lib:$JAVA_HOME/jre/lib/amd64/server:/usr/lib/openmpi/lib
ENV CLASSPATH=$CLASSPATH:/opt/COMPSs/Runtime/compss-engine.jar

ARG COMPSS_TAG=stable
ARG COMPSS_CLI=true
ARG COMPSS_SKIP_TESTS=true
ARG COMPSS_MONITOR=false
ARG COMPSS_STREAM=false
ARG COMPSS_JACOCO=false
ARG COMPSS_KAFKA=false

# build COMPSs
COPY build-compss.sh /
RUN bash -x /build-compss.sh $COMPSS_TAG $COMPSS_BINDING_C $COMPSS_BINDING_PYTHON $COMPSS_EXTRAE $COMPSS_AUTOPARALLEL $COMPSS_SKIP_TESTS $COMPSS_MONITOR $COMPSS_STREAM $COMPSS_JACOCO $COMPSS_KAFKA $COMPSS_CLI

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]