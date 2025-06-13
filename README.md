# Cat Activities Monitor

## Requirements

- Camera that provides unauthenticated real-time snapshots available at an acessible URL.
- A machine (computer, raspberry pi, any iot device) that can run the api and frontend.

## Setup

### Docker-Compose

1 - The docker-compose.example.yml file contains a base configuration for the docker compose. You can use it as a reference to create your own, or just copy and paste it.
2 - Change the <your-server-ip-address> to the ip address / domain name of the machine that will run the api and frontend.
3 - Create a config.yaml file next to your docker-compose.yml file.
4 - Use the config.yaml.example file as a reference to create your own.
5 - Run the docker compose up command.
6 - Go to <your-server-ip-address>:3000 and you should see the frontend <3

## Dev Setup

This project is composed of two main parts: api and frontend.

Each folder has a README.md file with instructions on how to run the project.
You can also run the project using the docker compose file in the root of the project.

