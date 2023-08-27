# syntax=docker/dockerfile:1
FROM python:3.11

ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /code
ADD ./requirements.txt /code/requirements.txt
RUN pip install -r requirements.txt

# Creating folders, and files for a project:
COPY . /code

# Start the application:
ADD docker-entrypoint.sh /code/docker-entrypoint.sh
RUN chmod a+x /code/docker-entrypoint.sh
ENTRYPOINT ["/code/docker-entrypoint.sh"]