echo [$(date)]: "START"

echo [$(date)]: "Creating env with Python 3.8 v"

conda create --prefix ./env python=3.8 -y

echo [$(date)]: "Activating the environment"

source activate ./env

echo [$(date)] : "installing dev requirements"

pip install -r requirements_dev.txt

echo [$(date)]: "END"