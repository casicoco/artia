# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* artia/*.py

black:
	@black scripts/* artia/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr artia-*.dist-info
	@rm -fr artia.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

# ----------------------------------
#      GCP
# ----------------------------------
GCE_INSTANCE_NAME="instance-data"
GCP_PROJECT_ID="airtia"
GCE_ZONE="europe-west1-b"

set_project:
	gcloud config set project ${GCP_PROJECT_ID}

create_compute:
	gcloud compute instances create ${GCE_INSTANCE_NAME} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE} --machine-type=e2-medium

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
	LOCAL_PATH = '/Users/lauredegrave/code/casicoco/TaxiFareModel/TaxiFareModel/data/test.csv' # path to the file to upload to GCP (the path to the file should be absolute or should match the directory where the make command is ran)
	BUCKET_FOLDER = data # bucket directory in which to store the uploaded file (`data` is an arbitrary name that we choose to use)
	BUCKET_FILE_NAME = $(shell basename ${LOCAL_PATH}) # name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \ #taxi_train_pipeline_$(shell date +'%Y%m%d_%H%M%S')
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \ #'trainings'
		--package-path ${PACKAGE_NAME} \ #TaxiFareModel
		--module-name ${PACKAGE_NAME}.${FILENAME} \ #trainer
		--python-version=${PYTHON_VERSION} \ #3.7
		--runtime-version=${RUNTIME_VERSION} \ #2.6
		--region ${REGION} \ #europe-west1
		--stream-logs \
    --scale-tier CUSTOM \
    --master-machine-type n1-standard-16

start_compute:
	gcloud compute instances start ${GCE_INSTANCE_NAME} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}

stop_compute:
	gcloud compute instances stop ${GCE_INSTANCE_NAME} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}

connect_ssh:
	gcloud beta compute ssh ${GCE_INSTANCE_NAME} --project ${GCP_PROJECT_ID} --zone ${GCE_ZONE}


# ----------------------------------
#      API
# ----------------------------------
run_api:
	uvicorn artia.app:app --reload

# ----------------------------------
#      DOCKER
# ----------------------------------
docker_init:
	export DOCKER_IMAGE_NAME="name-of-my-image-in-kebab-case"


# ----------------------------------
#      Streamlit
# ----------------------------------
run_streamlit:
	-@streamlit run app.py
