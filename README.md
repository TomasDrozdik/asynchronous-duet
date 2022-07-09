# Asynchronous Duet Benchmarking

Tool for running Asynchronous Duet Benchmarks.

This is a practical part of a [Master Thesis](https://github.com/TomasDrozdik/asynchronous-duet-thesis).

Documentation is on the [GitHub Wiki](https://github.com/TomasDrozdik/asynchronous-duet/wiki)

## Install/Buid/Run TL;DR

```
# Install duet tools
python3 -m venv venv
source venv/bin/activate
pip install .

# Build docker images, WARNING: need git LFS
git lfs checkout
# Provide zipped speccpu to ./benchmarks/speccpu/speccpu.zip
./build.sh -d <docker|podman>

# Make sure following docker images exist: ["barrier-agent", "renaissance", "dacapo", "scalabench", "speccpu"]

# Run test configs, if result dir exists, command fails
duetbench --verbose --outdir results.test --docker <podman|docker> ./benchmarks/renaissance/test.duet.yml ./benchmarks/dacapo/test.duet.yml ./benchmarks/scalabench/test.duet.yml ./benchmarks/speccpu/test.duet.yml 2>&1 | tee results.test.log

# Run actual runs see [Runs iterations and timeouts](#runs-iterations-and-timeouts)
duetbench --verbose --outdir results --docker <podman|docker> ./benchmarks/renaissance/duet.yml ./benchmarks/dacapo/duet.yml ./benchmarks/scalabench/duet.yml ./benchmarks/speccpu/duet.yml 2>&1 | tee results.log
```
