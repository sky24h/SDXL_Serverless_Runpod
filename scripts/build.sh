docker_image_name=sky24/t2i:release
docker build . --tag=$docker_image_name
docker push $docker_image_name
