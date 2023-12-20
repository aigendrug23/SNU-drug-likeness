FROM python:3.9.18

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY . /usr/src/app/
RUN python3.9 -m pip install --upgrade pip

# INSTALL conda
ENV PATH /opt/conda/bin:$PATH
# Install Dependencies of Miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean --all -y && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Refresh PATH and activate conda environment
SHELL ["/bin/bash", "--login", "-c"]
RUN exec bash && \
    . /root/.bashrc && \
    conda init bash
RUN conda config --append channels conda-forge && \
    conda create --name my_env --file packagelist.txt

SHELL [ "/bin/bash", "--login", "-c" ]
# RUN echo "conda activate my_env" >> ~/.bashrc
RUN conda activate my_env && \
    pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r requirements.txt


EXPOSE 5000
# ENTRYPOINT ["conda", "run", "-n", "my_env", "python", "-m", "flask", "run", "--host=0.0.0.0"]
ENTRYPOINT bash entrypoint.sh
