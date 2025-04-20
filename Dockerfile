# 1. Choose the official Ubuntu image
FROM ubuntu:latest

# 2. Update package lists and install Python 3.11 and Docker
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.11-venv

# 3. Set the working directory in the container
WORKDIR /app

# 4. Copy the requirements.txt file to the container
COPY requirements.txt /app/

# 5. Install Python dependencies
RUN python3.11 -m pip install --upgrade --no-cache-dir -r requirements.txt

# 6. Authenticate with GitHub Container Registry

# 7. Copy all application files to the container
COPY . /app/

# 8. Set the command to run the server (example for a FastAPI application)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
