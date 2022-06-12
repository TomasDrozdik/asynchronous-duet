#!/bin/bash

set -e

git clone https://github.com/d-iii-s/java-ubench-agent.git
pushd java-ubench-agent
ant
popd
