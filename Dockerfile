FROM anibali/pytorch:latest

ARG model_files=/app
ARG config_file=config.pt
ARG model_parameter_file=air_force_model.pt
ARG model_hyper_file=af_hyper.pt
ARG model_cat_file=af_cat_encoder.pt
ARG model_cont_file=af_continuous_encoder.pt
ARG port=5000
ARG host

ENV AF_PORT=$port
ENV AF_HOST=$host
ENV AF_MODEL_FILES=$model_files

RUN conda install -y flask \
    && conda install -c pytorch torchvision \
    && conda install waitress
RUN conda install pandas
RUN conda install -c conda-forge category_encoders onnx matplotlib vim
RUN conda install -c anaconda pymongo
RUN mkdir -p /app

COPY ./air_force_shared.py /app
COPY ./air_force_model_server.py /app
COPY ./air_force_nn.py /app
COPY ./FY11Leads.csv /app
COPY ./TestPreAuthData.csv /app

COPY ./$config_file /app/
COPY ./$model_parameter_file /app/
COPY ./$model_hyper_file /app/
COPY ./$model_cat_file /app/
COPY ./$model_cont_file /app/

COPY ./run-model-service.sh /

EXPOSE $port

# Allow shell to be run from docker run so that
# training can be done in the image from command line.
#
CMD ["/run-model-service.sh"]
