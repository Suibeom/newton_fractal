az disk create --resource-group MC_art_moecluster_eastus --name myAKSDisk --size-gb 20 --query id --output tsv --sku Standard_LRS
az aks update -n moecluster -g art --attach-acr fractalart
az aks get-credentials -g art -n moecluster
az acr build --image art/newton_fractal --registry fractalart /Users/Suibeom/PycharmProjects/newton_fractals/newton_fractals/
kubectl apply -f deploy.yaml