FROM ubuntu:jammy-20221130

COPY .github/texlive /
RUN apt-get update -y; DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC  apt-get install -y $(cat /requirements.txt)

COPY .github/python /
RUN  apt-get install -y python3 python3-pip python-is-python3; pip install -r /requirements.txt

COPY Gemfile /
RUN apt-get install -y ruby-full; gem install bundler; bundle check || bundle install --jobs=4 --retry=3; rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
