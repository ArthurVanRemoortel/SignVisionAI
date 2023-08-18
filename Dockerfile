# syntax=docker/dockerfile:1
FROM python:3.11

# Set environment variables
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Configure Poetry
ENV POETRY_VERSION=1.5.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

# Set the working directory
WORKDIR /code

# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml /code/

# Project initialization:
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Creating folders, and files for a project:
COPY . /code

ADD docker-entrypoint.sh /code/docker-entrypoint.sh
RUN chmod a+x /code/docker-entrypoint.sh
ENTRYPOINT ["/code/docker-entrypoint.sh"]