FROM python:3.12-bullseye

RUN pip install -U pip

COPY . /

RUN pip install -r ./requirements.txt --upgrade

RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

RUN chmod +x ./startPoolWorkers.sh

CMD [ "/startPoolWorkers.sh" ]
