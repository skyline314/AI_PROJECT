# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]