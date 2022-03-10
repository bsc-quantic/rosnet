FROM compss/compss:latest

# install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa; \
	apt update; \
	DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-dev python3.9-distutils; \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1; \
	update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2

# install Python dependencies
RUN python3.9 -m pip install --upgrade pip setuptools; \
	python3.9 -m pip install numpy autoray multimethod dill pycodestyle;

# build COMPSs from source
RUN rm -r /opt/COMPSs; \
	git clone https://github.com/bsc-wdc/compss.git /compss; \
	cd /compss; \
	./submodules_get.sh; ./submodules_patch.sh; \
	cd builders; ./buildlocal --no-monitor --pycompss --no-tracing --no-autoparallel --no-kafka --no-jacoco --cli --skip-tests /opt/COMPSs/