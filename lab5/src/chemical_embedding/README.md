# Interactive Chemical Embeddings

This is an application that can be used to interactively explore chemical embeddings

## Quickstart
To facilitate easy installation, this is destributed as a Docker image


### Docker

1. First install docker and start the daemon

2. Build the container from this directory

    docker build -t chemical_embedding .
	
3. Run the container from this directory 

	docker run -p 5006:5006 chemical_embedding

Then navigate to [http://0.0.0.0:5006/app](http://0.0.0.0:5006/app) in the browser


### Pip install

Install packages
 
    pip install --no-cache-dir -r requirements.txt

Start the server

    panel serve app.py --address 0.0.0.0 --port 5006 --allow-websocket-origin="*"

Then navigate to [http://0.0.0.0:5006/app](http://0.0.0.0:5006/app) in the browser
