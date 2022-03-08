FROM openjdk:8-jdk

# install COMPSSs dependencies
RUN apt-get update; apt-get install -y graphviz xdg-utils libtool automake build-essential python python-dev libpython2.7 python3 python3-dev libboost-serialization-dev libboost-iostreams-dev  libxml2 libxml2-dev csh gfortran libgmp3-dev flex bison texinfo python3-pip libpapi-dev ssh

# install gradle
ADD https://services.gradle.org/distributions/gradle-5.4.1-bin.zip /opt
RUN unzip /opt/gradle-5.4.1-bin.zip -d /opt

# set environment
ENV GRADLE_HOME="/opt/gradle-5.4.1"
ENV PATH="/opt/gradle-5.4.1/bin:${PATH}"

# install pycompss
RUN pip3 install pycompss