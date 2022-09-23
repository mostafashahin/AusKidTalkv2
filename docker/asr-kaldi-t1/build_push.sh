
set -e

docker build -t asrt1 --no-cache .

docker tag asrt1:latest mostafaashahin/auskidtalk:asrt1
docker login -u mostafaashahin -p ca4b6c91-3278-4254-b77b-612e2768dced
docker push mostafaashahin/auskidtalk:asrt1
