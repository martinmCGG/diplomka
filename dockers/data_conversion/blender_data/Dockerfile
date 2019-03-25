# Start with ubuntu base image
FROM ubuntu:xenial

# Install basic utilities, python and blender prerequisiteis
RUN apt-get update && apt-get install -yq \
		build-essential \
		curl \
		bzip2 \
		libfreetype6 \
		libgl1-mesa-dev \
		libglu1-mesa \
		libxi6 \
		libxrender1 \
		python2.7  && \
	apt-get -y autoremove && \
	rm -rf /var/lib/apt/lists/*

# Set blender version 
ENV BLENDER_MAJOR 2.79
ENV BLENDER_VERSION 2.79
ENV BLENDER_BZ2_URL https://mirror.clarkson.edu/blender/release/Blender$BLENDER_MAJOR/blender-$BLENDER_VERSION-linux-glibc219-x86_64.tar.bz2

# Download and install blender
RUN mkdir /usr/local/blender && \
	curl -SL "$BLENDER_BZ2_URL" -o blender.tar.bz2 && \
	tar -jxvf blender.tar.bz2 -C /usr/local/blender --strip-components=1 && \
	rm blender.tar.bz2

# Copy changed files
RUN mkdir /blender_scripts
WORKDIR /blender_scripts
RUN touch __init__.py
ENV PATH="/usr/local/blender/blender:${PATH}"

COPY *.blend ./
COPY *.py ./
COPY *.ini ./

