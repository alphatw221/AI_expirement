tensorflow 2.4
CUDA 11.0   對應 cuDNN 8.0
tensorflow 2.4 bug  => <CUDA Path>/extras/CUPTI/lib64/cupti_2020.1.0.dll 移動到 <CUDA Path>/bin/ 改名 "cupti64_110.dll"

tensorflow2.5
CUDA 11.2 對應 cuDNN 8.1
tensorflow 2.5 bug => <CUDA Path>/bin/cusolver64_10.dll 改名 "cusolver64_11.dll"

pip install virtualenv
python -m virtualenv --python=C:\Users\xxx\xxx <env_name>

<env_name>\Script\activate

pip install -r requirement.txt

deactivate
