# EAD-Net
Early detection of AD

Alzheimer's disease (AD) is the most prevalent neurodegenerative disorder among older adults. Early diagnosis, particularly the identification of Mild Cognitive Impairment (MCI), is crucial for effective AD management. This study proposes an early diagnosis network (EAD-Net) for the detection of AD using T1-weighted MRI (T1WI). EAD-Net incorporates a multi-scale wavelet transform (MWT) module to perform image denoising and generate high- and low-frequency subbands that preserve edge and detail information. Next, a layer alternating feature encoder based on Mamba and CNN is proposed to extract and optimize multi-level feature representations. Finally, we propose a class-independent multi-binary (CIR) classifier to enhance flexibility of EAD-Net and improve classification accuracy across different groups. 
This official code will be  publicly opened after the acceptance of the manuscript.

ADNI: The public dataset is available at https://adni.loni.usc.edu/data-samples/adni-data/

## Requirement
acvl_utils==0.2
antspyx==0.5.4
batchgenerators==0.25
batchgeneratorsv2==0.1.1
certifi==2024.7.4
charset-normalizer==3.3.2
connected-components-3d==3.17.0
contourpy==1.2.1
cycler==0.12.1
dicom2nifti==2.4.11
dynamic_network_architectures==0.3.1
einops==0.8.0
et-xmlfile==1.1.0
fft-conv-pytorch==1.2.0
filelock==3.15.4
fonttools==4.53.1
fsspec==2024.6.1
future==1.0.0
graphviz==0.20.3
huggingface-hub==0.24.5
idna==3.7
imagecodecs==2024.6.1
imageio==2.34.2
Jinja2==3.1.4
joblib==1.4.2
kan==0.0.2
kiwisolver==1.4.5
lazy_loader==0.4
linecache2==1.0.0
mamba-ssm==2.2.2
MarkupSafe==2.1.5
matplotlib==3.9.1
MedPy==0.5.1
mpmath==1.3.0
networkx==3.3
nibabel==5.2.1
ninja==1.11.1.1
numpy==1.26.4
opencv-python==4.10.0.84
openpyxl==3.1.5
packaging==24.1
pandas==2.2.2
patsy==1.0.1
pillow==10.4.0
pydicom==2.4.4
pyparsing==3.1.2
python-dateutil==2.9.0.post0
python-gdcm==3.0.24.1
pytz==2024.1
PyYAML==6.0.1
regex==2024.7.24
requests==2.32.3
safetensors==0.4.3
scikit-image==0.24.0
scikit-learn==1.5.1
scipy==1.14.0
seaborn==0.13.2
SimpleITK==2.3.1
six==1.16.0
statsmodels==0.14.4
sympy==1.13.0
threadpoolctl==3.5.0
tifffile==2024.7.2
timm==1.0.8
tokenizers==0.19.1
torch==2.3.1
torchvision==0.18.1
tqdm==4.66.4
traceback2==1.4.0
transformers==4.43.3
triton==2.3.1
typing_extensions==4.12.2
tzdata==2024.1
unittest2==1.1.0
urllib3==2.2.2
webcolors==24.11.1
yacs==0.1.8
