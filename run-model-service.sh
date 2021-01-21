#!/bin/bash
# run-model-service.sh
cd /app

# create_app() is a function in air_force_model_server.py, where the flask
# app and its routes are initialized.
#
# To do this from a command line, use:
# $ FLASK_ENV=development FLASK_APP=air_force_model_server.py flask run

#
waitress-serve --port=${AF_PORT} --call 'air_force_model_server:create_app'
