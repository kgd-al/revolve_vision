#!/bin/bash

user=kgd
host=karine_ripper
base=$user@$host:code

update(){
  rsync -avzhP --update --prune-empty-dirs $@
}

update bin src $base/revolve_vision
update ../abrain/src $base/abrain/
update ../revolve/{core,runners} $base/revolve2/
