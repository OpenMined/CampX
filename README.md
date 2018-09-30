# CampX

Agent Training environments that are entirely created using tensor operations. The API is a modified port of [PyColab](https://github.com/deepmind/pycolab) and should be loosely compatible with worlds created for PyColab.

#### Cloning:
In order to get everything, use git's `--recursive` flag:
```bash
git clone --recursive https://github.com/OpenMined/CampX.git
```
This will also download the [`safe-grid-agents`](https://github.com/jvmancuso/safe-grid-agents) and [`ai-safety-gridworlds`](https://github.com/deepmind/ai-safety-gridworlds) repos, which can be used for training fast plaintext agents in safety-minded gridworlds.  See those repos' respective READMEs for installation/usage instructions.

### Install

```sh
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup.py install # installs campx from latest git build
python -m ipykernel install --user --name=campx
jupyter notebook # select campx kernel
```