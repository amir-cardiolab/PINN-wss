# PINN-wss

Using physics-informed neural networks (PINN) to compute wall shear stress (WSS) from sparse data and without knowledge of boundary conditions. \

Codes and data used in the examples presented in the paper: \
Uncovering near-wall blood flow from sparse data with physics-informed neural networks \
https://arxiv.org/abs/2104.08249  \
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Pytorch codes are included for the different examples presented in the paper. Namely, 1D advection-diffusion, 2D stenosis, 2D aneurysm, 3D aneurysm, and parameter identification (viscosity).  \
Sample code is included for converting Pytorch output to VTK format that could be visualized in ParaView. The neural networks used in the Torch2VTK codes need to be compatible with the neural networks that are saved by the PINN code.  \

%%%%%%%%%
Data: \
The input data for the 2D cases are provided in the Data folder. For the 3D model, a Google Drive link to the 3D files is included. \

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Installation: \
Install Pytorch: \
https://pytorch.org/

Install VTK after Pytorch is installed.  \
An example with pip:

conda activate pytorch \
pip install vtk 
