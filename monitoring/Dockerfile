FROM python:3.11-slim

RUN pip install -U pip

COPY . /

RUN pip install -r ./requirements.txt --upgrade

RUN prefect config set PREFECT_API_URL=http://orchestration:4200/api

RUN chmod +x ./startPoolWorkers.sh

CMD [ "bash", "./startPoolWorkers.sh" ]