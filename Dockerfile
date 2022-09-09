FROM ubuntu:20.04

run DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get update && apt-get install -y \
    python3.4 \
    python3-pip \
    git

ADD requirements.txt .

RUN python3 --version

RUN pip3 install -r requirements.txt

ADD . .

CMD streamlit run ./main.py --server.maxUploadSize 5000