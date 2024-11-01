# FIXME: Choose an appropriate base image 
# Using a lightweight python:3.10-slim image to reduce the size of the Docker container.
FROM python:3.10-slim  

# Set the working directory inside the container
WORKDIR /app

# FIXME: Copy necessary files and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# FIXME: Set up the command to run your FastAPI application
# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


