# sudo docker run -it --rm --mount type=bind,src=/home/kardosp/docker/OnlineGen/Player,dst=/unity_player -p 12583:12583 syntehtic-image-generator bash
docker run -idt --rm --mount type=bind,src=<REPLACE_PATH_TO_GENERATOR>,dst=/unity_player -p 12583:12583 --name synthetic-image-generator synthetic-image-generator bash -c "screen ./start.sh"
