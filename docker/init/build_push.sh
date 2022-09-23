
set -e

docker build --no-cache -t init:latest .

docker tag init:latest mostafaashahin/auskidtalk:init
docker login -u mostafaashahin -p ca4b6c91-3278-4254-b77b-612e2768dced

docker push mostafaashahin/auskidtalk:init
