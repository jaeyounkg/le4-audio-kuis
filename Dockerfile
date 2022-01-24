FROM python:3.9.9-slim-buster

RUN apt-get update \
 && apt-get install --no-install-recommends -y fonts-ipaexfont libglib2.0-0 git gcc wget \
# ref: https://github.com/ros-planning/navigation/issues/579#issuecomment-307590262
 && apt-get install --no-install-recommends -y libsdl-image1.2-dev libsdl-dev libsdl-mixer1.2-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/local/work
COPY requirements.lock .
RUN pip install -U pip
RUN pip install numpy==1.20.3 scipy==1.7.3 seaborn==0.11.2 librosa==0.8.1
RUN pip install Kivy==2.0.0
RUN pip install Kivy-Garden==0.1.4
RUN pip install matplotlib==3.1.3
RUN garden install matplotlib


CMD ["python", "app/karaoke_app.py"]
