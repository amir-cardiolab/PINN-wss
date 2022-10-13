# PINN-wss

Using physics-informed neural networks (PINN) to compute wall shear stress (WSS) from sparse data and without knowledge of boundary conditions. 

Codes and data used in the examples presented in the paper: \
Uncovering near-wall blood flow from sparse data with physics-informed neural networks \
https://arxiv.org/abs/2104.08249  \
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Pytorch codes are included for the different examples presented in the paper. Namely, 1D advection-diffusion, 2D stenosis, 2D aneurysm, 3D aneurysm, and parameter identification (viscosity).  \
Sample code is included for converting Pytorch output to VTK format that could be visualized in ParaView. The neural networks used in the Torch2VTK codes need to be compatible with the neural networks that are saved by the PINN code.\
IMPORTANT NOTE on visualizing the results: The input coordiantes need to be normalized but then mapped back to the physical coordinates for visualization.  You need to load the physical mesh and the saved PINN networks. Then you read the coordinates of the physical mesh (x,y). For each node you scale it by X_scale and Y_ cale and then you feed that scaled coordinates into the loaded PINN network to produce velocity for that node and then assign that velocity to that node in the "original" physical mesh to visualize it. 


%%%%%%%%% \
Data: \
The input data for the 2D cases are provided in the Data folder. For the 3D model, a Google Drive link to the 3D files is included. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Installation: \
Install Pytorch: \
https://pytorch.org/

Install VTK after Pytorch is installed.  \
An example with pip:

conda activate pytorch \
pip install vtk 
