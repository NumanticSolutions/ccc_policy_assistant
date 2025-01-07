# [2501]
#

#
# make all PROJECT=the.project.name DATASET=dataset.name
#

buckets :
	@echo "GCP : BUCKETS"
	gcloud storage ls
	@echo ""

containers :
	@echo "GCP : CONTAINERS"
	gcloud container images list
	@echo ""

datasets :
	@echo "GCP : DATASETS - BIGQUERY"
	bq show
	@echo ""

describe :
	@echo "GCP : DESCRIBE"
	gcloud projects describe $(PROJECT)
	@echo ""

info :
	@echo "GCP : INFO"
	gcloud compute project-info describe --flatten="commonInstanceMetadata[]"
	# gcloud compute project-info describe
	@echo ""

projects :
	@echo "GCP : PROJECTS"
	gcloud projects list
	@echo ""

services :
	@echo "GCP : SERVICES"
	gcloud run services list
	@echo ""

tables :
	@echo "GCP : TABLES - IN DATASET"
	bq ls --max_results 1000 '$(PROJECT):$(DATASET)'
	@echo ""

all : buckets containers datasets describe info projects services tables


# [ ] access to billing data is currently only in beta?
# [ ] list of services currently in use by VertexAI?
# [ ] some sort of usage data (not derived from conventional analytics)?
#

