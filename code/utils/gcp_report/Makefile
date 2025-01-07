# [2501]
#

#
# * https://gist.github.com/pydevops/cffbd3c694d599c6ca18342d3625af97
#

buckets :
	@echo "GCP BUCKETS"
	gcloud storage ls
	@echo ""

containers :
	@echo "GCP CONTAINERS"
	gcloud container images list
	@echo ""

datasets :
	@echo "GCP DATASETS"
	bq show
	@echo ""

describe :
	@echo "GCP DESCRIBE PROJECT"
	gcloud projects describe eternal-bongo-435614-b9
	@echo ""

metadata :
	@echo "GCP METADATA"
	gcloud compute project-info describe --flatten="commonInstanceMetadata[]"
	# gcloud compute project-info describe
	@echo ""

projects :
	@echo "GCP PROJECTS"
	gcloud projects list
	@echo ""

services :
	@echo "GCP SERVICES" 
	gcloud run services list
	@echo ""

# for a given project and dataset
#
tables :
	@echo "GCP PROJECT DATASET TABLES" 
	bq ls --max_results 1000 'eternal-bongo-435614-b9:ccc_polasst'
	@echo


all : buckets containers datasets describe metadata projects services tables 


# [ ] list of services used by VertexAI?
# [ ] billing info : currently only in beta?
#

