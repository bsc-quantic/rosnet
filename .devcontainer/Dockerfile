FROM compss/compss:latest

RUN apt remove python3-pip python3-dev; \
	add-apt-repository ppa:deadsnakes/ppa; \
	apt update; \
	DEBIAN_FRONTEND=noninteractive apt install -y python3.9 python3.9-dev python3.9-distutils

# # install pycompss
RUN python3.9 -m pip install --upgrade pip setuptools; \
	python3.9 -m pip install pycompss; \
	python3.9 -m pip install numpy autoray multimethod pytest pytest-cov;