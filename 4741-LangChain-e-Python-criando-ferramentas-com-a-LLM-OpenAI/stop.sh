#!/bin/bash
docker stop app_langchain_container
docker rm app_langchain_container
docker rmi app_langchain
echo "Container e imagem removidos com sucesso para liberar espaço."
