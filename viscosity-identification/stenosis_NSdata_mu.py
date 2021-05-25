import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import vtk
from vtk.util import numpy_support as VN
#import torch.optim.lr_scheduler.StepLR





def geo_train(device,x_in,y_in,xb,yb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt,T,xb_inlet,yb_inlet,ub_inlet,vb_inlet ):
	if (Flag_batch):
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device)
	 #dataset = TensorDataset(x,y)
	 xb = torch.Tensor(xb).to(device)
	 yb = torch.Tensor(yb).to(device)
	 ub = torch.Tensor(ub).to(device)
	 vb = torch.Tensor(vb).to(device)
	 xd = torch.Tensor(xd).to(device)
	 yd = torch.Tensor(yd).to(device)
	 ud = torch.Tensor(ud).to(device)
	 vd = torch.Tensor(vd).to(device)
	 #dist = torch.Tensor(dist).to(device)
	 xb_inlet = torch.Tensor(xb_inlet).to(device)
	 yb_inlet = torch.Tensor(yb_inlet).to(device)
	 ub_inlet = torch.Tensor(ub_inlet).to(device)
	 vb_inlet = torch.Tensor(vb_inlet).to(device)
	 if(1): #Cuda slower in double? 
		 x = x.type(torch.cuda.FloatTensor)
		 y = y.type(torch.cuda.FloatTensor)
		 xb = xb.type(torch.cuda.FloatTensor)
		 yb = yb.type(torch.cuda.FloatTensor)
		 ub = ub.type(torch.cuda.FloatTensor)
		 vb = vb.type(torch.cuda.FloatTensor)
		 #dist = dist.type(torch.cuda.FloatTensor)
		 xb_inlet = xb_inlet.type(torch.cuda.FloatTensor)
		 yb_inlet = yb_inlet.type(torch.cuda.FloatTensor)
		 ub_inlet = ub_inlet.type(torch.cuda.FloatTensor)
		 vb_inlet = vb_inlet.type(torch.cuda.FloatTensor)
		 xd = xd.type(torch.cuda.FloatTensor)
		 yd = yd.type(torch.cuda.FloatTensor)
		 ud = ud.type(torch.cuda.FloatTensor)
		 vd = vd.type(torch.cuda.FloatTensor)


	 dataset = TensorDataset(x,y)
	 #dataset_bc = TensorDataset(x,y,xb,yb,ub,vb,dist)

	 #dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = True )
	 #dataloader_bc = DataLoader(dataset_bc, batch_size=batchsize,shuffle=True,num_workers = 0, drop_last = False )
	else:
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device) 
	 #t = torch.Tensor(t_in).to(device) 
	#x_test =  torch.Tensor(x_test).to(device)
	#y_test  = torch.Tensor(y_test).to(device)  

	h_n = 128 #Width for u,v,p
	input_n = 2 # this is what our answer is a function of (x,y)
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class MySquared(nn.Module):
		def __init__(self, inplace=True):
			super(MySquared, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			return torch.square(x)

	class Mu_visc(nn.Module):
		def __init__(self):
			super(Mu_visc, self).__init__()
			#self.multp = Variable(torch.rand(1).to(device), requires_grad=True)
			self.multp = Variable(torch.abs(torch.rand(1)).to(device), requires_grad=True) #does not seem to enforce positive
		def forward(self):
			#return self.multp
			return  abs(self.multp)

	

	class Net2_u(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_u(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(y- yStart) * (y- yEnd ) + U_BC_in + (y- yStart) * (y- yEnd )  #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_v(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_v(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_p(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_p, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			#print('shape of xnet',x.shape) #Resuklts: shape of xnet torch.Size([batchsize, 2]) 
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  (1-x[:,0]) * output[:,0]  #Enforce P=0 at x=1 #Shape of output torch.Size([batchsize, 1])
			return  output
	
	################################################################
	#net1 = Net1().to(device)
	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_p = Net2_p().to(device)
	mu_visc = Mu_visc().to(device)

	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2_u.apply(init_normal)
	net2_v.apply(init_normal)
	net2_p.apply(init_normal)


	############################################################################

	optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	learning_rate_mu = learning_rate * lr_factor_mu
	optimizer_mu = optim.Adam([{'params':[mu_visc.multp], 'lr':learning_rate_mu}], betas = (0.9,0.99),eps = 10**-15)




	def criterion(x,y):

		#print (x)
		#x = torch.Tensor(x).to(device)
		#y = torch.Tensor(y).to(device)
		#t = torch.Tensor(t).to(device)

		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		y.requires_grad = True
		#t.requires_grad = True
		#u0 = u0.detach()
		#v0 = v0.detach()
		

		net_in = torch.cat((x,y),1)
		u = net2_u(net_in)
		u = u.view(len(u),-1)
		v = net2_v(net_in)
		v = v.view(len(v),-1)
		P = net2_p(net_in)
		P = P.view(len(P),-1)



		mu_v = mu_visc()

		
		u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		#u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		#v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		
		XX_scale = U_scale * (X_scale**2)
		YY_scale = U_scale * (Y_scale**2)
		UU_scale  = U_scale **2
	
		loss_2 = u*u_x / X_scale + v*u_y / Y_scale - mu_v*( u_xx/XX_scale  + u_yy /YY_scale  )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
		loss_1 = u*v_x / X_scale + v*v_y / Y_scale - mu_v*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
		loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity



		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1))+  loss_f(loss_2,torch.zeros_like(loss_2))+  loss_f(loss_3,torch.zeros_like(loss_3))

		return loss

	def Loss_BC(xb,yb,ub,vb, xb_inlet, yb_inlet, ub_inlet, x, y ):
		#Stream function 
		if(0):
			xb = torch.FloatTensor(xb).to(device)
			yb = torch.FloatTensor(yb).to(device)
			ub = torch.FloatTensor(ub).to(device)
		

		net_in1 = torch.cat((xb, yb), 1)
		out1_u = net2_u(net_in1)
		out1_v = net2_v(net_in1)
		
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)

		net_in2 = torch.cat((xb_inlet, yb_inlet), 1)
		out2_u = net2_u(net_in2)
		out2_v = net2_v(net_in2)
		
		out2_u = out2_u.view(len(out2_u), -1)
		out2_v = out2_v.view(len(out2_v), -1)

	

		loss_f = nn.MSELoss()
		loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + loss_f(out1_v, torch.zeros_like(out1_v)) 
		loss_inlet = loss_f(out2_u, ub_inlet) + loss_f(out2_v, torch.zeros_like(out2_v) )

		return 1.* loss_noslip + loss_inlet
		#return loss_noslip


	def Loss_data(xd,yd,ud,vd ):
	

		#xb.requires_grad = True
		#xd.requires_grad = True
		#yd.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		net_in1 = torch.cat((xd, yd), 1)
		out1_u = net2_u(net_in1)
		out1_v = net2_v(net_in1)
		
		out1_u = out1_u.view(len(out1_u), -1)
		out1_v = out1_v.view(len(out1_v), -1)

	

		loss_f = nn.MSELoss()
		loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) 


		return loss_d

	# Main loop

	tic = time.time()


	if(Flag_pretrain):
		print('Reading (pretrain) functions first...')
		net2_u.load_state_dict(torch.load(path+"sten_u" + ".pt"))
		net2_v.load_state_dict(torch.load(path+"sten_v" + ".pt"))
		net2_p.load_state_dict(torch.load(path+"sten_p" + ".pt"))
	
		

	if (Flag_schedule):
		scheduler_u = torch.optim.lr_scheduler.StepLR(optimizer_u, step_size=step_epoch, gamma=decay_rate)
		scheduler_v = torch.optim.lr_scheduler.StepLR(optimizer_v, step_size=step_epoch, gamma=decay_rate)
		scheduler_p = torch.optim.lr_scheduler.StepLR(optimizer_p, step_size=step_epoch, gamma=decay_rate)
		scheduler_mu = torch.optim.lr_scheduler.StepLR(optimizer_mu, step_size=step_epoch, gamma=decay_rate)

	if(Flag_batch):# This one uses dataloader
			
			for epoch in range(epochs):
				#for batch_idx, (x_in,y_in) in enumerate(dataloader):  
				#for batch_idx, (x_in,y_in,xb_in,yb_in,ub_in,vb_in) in enumerate(dataloader): 
				loss_eqn_tot = 0.
				loss_bc_tot = 0.
				loss_data_tot = 0.
				n = 0
				for batch_idx, (x_in,y_in) in enumerate(dataloader): 
					#net_in = torch.cat((x_in,y_in),1)
					#u_bc = net1_bc_u(net_in)
					#v_bc = net1_bc_v(net_in)
					#dist_bc = net1_dist(net_in)

					#net2_psi.zero_grad()
					net2_u.zero_grad()
					net2_v.zero_grad()
					net2_p.zero_grad()
					mu_visc.zero_grad()
					loss_eqn = criterion(x_in,y_in)
					loss_bc = Loss_BC(xb,yb,ub,vb,xb_inlet,yb_inlet,ub_inlet,x,y)
					loss_data = Loss_data(xd,yd,ud,vd)
					loss = loss_eqn + Lambda_BC* loss_bc + Lambda_data*loss_data
					loss.backward()
					optimizer_u.step() 
					optimizer_v.step()
					#optimizer_psi.step()  
					optimizer_p.step()  
					optimizer_mu.step()
					loss_eqn_tot += loss_eqn
					loss_bc_tot += loss_bc
					loss_data_tot  += loss_data
					n += 1 
					if batch_idx % 40 ==0:
						#loss_bc = Loss_BC(xb,yb,ub,vb) #causes out of memory issue for large data in cuda
						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} Loss BC {:.8f} Loss data {:.8f}'.format(
							epoch, batch_idx * len(x_in), len(dataloader.dataset),
							100. * batch_idx / len(dataloader), loss_eqn.item(), loss_bc.item(),loss_data.item()))
						#print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss eqn {:.10f} '.format(
						#	epoch, batch_idx * len(x_in), len(dataloader.dataset),
						#	100. * batch_idx / len(dataloader), loss.item()))
				if (Flag_schedule):
						scheduler_u.step()
						scheduler_v.step()
						scheduler_p.step()
						scheduler_mu.step()
				loss_eqn_tot = loss_eqn_tot / n
				loss_bc_tot = loss_bc_tot / n
				loss_data_tot = loss_data_tot / n
				print('*****Total avg Loss : Loss eqn {:.10f} Loss BC {:.10f} Loss data {:.10f} ****'.format(loss_eqn_tot, loss_bc_tot,loss_data_tot) )
				print('learned viscossity is {:.7f} '.format(mu_visc().item()))
				print('learning rate V, mu :  ', optimizer_u.param_groups[0]['lr'], optimizer_mu.param_groups[0]['lr'])
		
			if(0): #This causes out of memory in cuda in autodiff
				loss_eqn = criterion(x,y)	
				loss_bc = Loss_BC(xb,yb,ub,vb)
				loss = loss_eqn #+ Lambda_BC* loss_bc
				print('**** Final (all batches) \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					loss.item(),loss_bc.item()))

	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			loss_eqn = criterion(x,y)
			loss_bc = Loss_BC(xb,yb,cb)
			if (Flag_BC_exact):
				loss = loss_eqn #+ loss_bc
			else:
				loss = loss_eqn + Lambda_BC * loss_bc
			loss.backward()
	
			optimizer_u.step() 
			optimizer_v.step() 
			optimizer_p.step() 
			if epoch % 10 ==0:
				print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					epoch, loss.item(),loss_bc.item()))

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
	#plot
	if(1):#save network
		torch.save(net2_p.state_dict(),path+"sten_dataMu_p" + ".pt")
		torch.save(net2_u.state_dict(),path+"sten_dataMu_u" + ".pt")
		torch.save(net2_v.state_dict(),path+"sten_dataMu_v" + ".pt")
		torch.save(mu_visc.state_dict(),path+"sten_dataMu_mu" + ".pt")

		print ("Data saved!")


	
	net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
	output_u = net2_u(net_in)  #evaluate model (runs out of memory for large GPU problems!)
	output_v = net2_v(net_in)  #evaluate model


	output_u = output_u.cpu().data.numpy() #need to convert to cpu before converting to numpy
	output_v = output_v.cpu().data.numpy()
	x = x.cpu()
	y = y.cpu()




	plt.figure()
	plt.subplot(2, 1, 1)
	plt.scatter(x.detach().numpy(), y.detach().numpy(), c = output_u , cmap = 'rainbow')
	plt.title('NN results, u')
	plt.colorbar()
	plt.show()
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.scatter(x.detach().numpy(), y.detach().numpy(), c = output_v , cmap = 'rainbow')
	plt.title('NN results, v')
	plt.colorbar()
	plt.show()



	

	return 

	############################################################
	##save loss
	##myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	##with myFile:
		#writer = csv.writer(myFile)
		#writer.writerows(LOSS)
	#LOSS = np.array(LOSS)
	#np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################



#######################################################
#Main code:
#device = torch.device("cpu")
device = torch.device("cuda")


Flag_batch = True 
Flag_BC_exact = False 
Lambda_BC  = 10.

Lambda_data = 1.

#Directory = "/home/aa3878/Data/ML/Amir/stenosis/"
Directory = "/scratch/aa3878/PINN/stenosis/PINN/"
mesh_file = Directory + "sten_mesh000000.vtu"
bc_file_in = Directory + "inlet_BC.vtk"
bc_file_wall = Directory + "wall_BC.vtk"

File_data = Directory + "velocity_sten_steady.vtu"
fieldname = 'f_5-0' #The velocity field name in the vtk file (see from ParaView)

batchsize = 256  
learning_rate = 1e-5 


epochs  = 5500  

Flag_pretrain = False # True #If true reads the nets from last run


Diff = 0.001
rho = 1.
T = 0.5 #total duraction
#nPt_time = 50 #number of time-steps

Flag_x_length = True #if True scales the eqn such that the length of the domain is = X_scale
X_scale = 2.0 #The length of the  domain (need longer length for separation region)
Y_scale = 1.0 
U_scale = 1.0
U_BC_in = 0.5


Lambda_div = 1.  #penalty factor for continuity eqn (Makes it worse!?)
Lambda_v = 1.  #penalty factor for y-momentum equation

#https://stackoverflow.com/questions/60050586/pytorch-change-the-learning-rate-based-on-number-of-epochs
Flag_schedule = True #If true change the learning rate 
if (Flag_schedule):
	learning_rate = 5e-4 #starting learning rate
	step_epoch = 1200 
	decay_rate = 0.1

lr_factor_mu = 0.1 # the learning rate for mu is multiplied by this

if (not Flag_x_length):
	X_scale = 1.
	Y_scale = 1.


print ('Loading', mesh_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))



nPt = 130  #400 #130
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 1.0
delta_circ = 0.2




t = np.linspace(0., T, nPt*nPt)
t=t.reshape(-1, 1)
print('shape of x',x.shape)
print('shape of y',y.shape)
#print('shape of t',t.shape)



## Define boundary points
print ('Loading', bc_file_in)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_in)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of at inlet' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_in  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh[:]),1)) 
yb_in  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh[:]),1))

print ('Loading', bc_file_wall)
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(bc_file_wall)
reader.Update()
data_vtk = reader.GetOutput()
n_pointsw = data_vtk.GetNumberOfPoints()
print ('n_points of at wall' ,n_pointsw)
x_vtk_mesh = np.zeros((n_pointsw,1))
y_vtk_mesh = np.zeros((n_pointsw,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_pointsw):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
xb_wall  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) 
yb_wall  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1))




#u_in_BC = np.linspace(U_BC_in, U_BC_in, n_points) #constant uniform BC
u_in_BC = (yb_in[:]) * ( 0.3 - yb_in[:] )  / 0.0225 * U_BC_in #parabolic


v_in_BC = np.linspace(0., 0., n_points)
u_wall_BC = np.linspace(0., 0., n_pointsw)
v_wall_BC = np.linspace(0., 0., n_pointsw)
#t_BC = np.linspace(0., T, nPt_BC)
#t_BC = np.linspace(0., T, nPt_time)

#t_BC = np.linspace(0., T, nPt_BC)
#t_BC = np.linspace(0., T, nPt_time)

#tb = np.concatenate((t_BC, t_BC, t_BC), 0)
#xb = np.concatenate((xb_wall), 0)
#yb = np.concatenate((yb_wall), 0)
xb = xb_wall
yb = yb_wall

#ub = np.concatenate((u_wall_BC), 0)
#vb = np.concatenate((v_wall_BC), 0)
ub = u_wall_BC
vb = v_wall_BC

#xb_inlet = np.concatenate((xb_in), 0)
#yb_inlet = np.concatenate((yb_in), 0)
#ub_inlet = np.concatenate((u_in_BC), 0)
#vb_inlet = np.concatenate((v_in_BC), 0)

xb_inlet = xb_in 
yb_inlet =yb_in 


ub_inlet = u_in_BC
vb_inlet = v_in_BC


### Trying to set distance function with Dirichlet BC everywhere
#xb_dist = np.concatenate((xleft, xup,xrightw, xdown,xdown2,xright), 0)
#yb_dist = np.concatenate((yleft, yup,yrightw, ydown,ydown2,yright), 0)
####


#tb= tb.reshape(-1, 1) #need to reshape to get 2D array
xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
ub= ub.reshape(-1, 1) #need to reshape to get 2D array
vb= vb.reshape(-1, 1) #need to reshape to get 2D array
xb_inlet= xb_inlet.reshape(-1, 1) #need to reshape to get 2D array
yb_inlet= yb_inlet.reshape(-1, 1) #need to reshape to get 2D array
ub_inlet= ub_inlet.reshape(-1, 1) #need to reshape to get 2D array
vb_inlet= vb_inlet.reshape(-1, 1) #need to reshape to get 2D array

print('shape of xb',xb.shape)
print('shape of yb',yb.shape)
print('shape of ub',ub.shape)


#V_IC = 0. #I.C. for all velocoties.
#t_IC = np.linspace(0., 0., nPt*nPt)
#u_IC = np.linspace(V_IC, V_IC, nPt*nPt)
#v_IC = np.linspace(V_IC, V_IC, nPt*nPt)
#t_IC= t_IC.reshape(-1, 1)
#u_IC= u_IC.reshape(-1, 1)
#v_IC= v_IC.reshape(-1, 1)


path = "Results/"



##### Read data here#########

#!!specify pts location here:
x_data = [1., 1.2, 1.22, 1.31, 1.39 ] 
y_data =[0.15, 0.07, 0.22, 0.036, 0.26 ]
z_data  = [0.,0.,0.,0.,0. ]

x_data = np.asarray(x_data)  #convert to numpy 
y_data = np.asarray(y_data) #convert to numpy 


print ('Loading', File_data)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(File_data)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the data file read:' ,n_points)


VTKpoints = vtk.vtkPoints()
for i in range(len(x_data)): 
	VTKpoints.InsertPoint(i, x_data[i] , y_data[i]  , z_data[i])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

probe = vtk.vtkProbeFilter()
probe.SetInputData(point_data)
probe.SetSourceData(data_vtk)
probe.Update()
array = probe.GetOutput().GetPointData().GetArray(fieldname)
data_vel = VN.vtk_to_numpy(array)



data_vel_u = data_vel[:,0] / U_scale
data_vel_v = data_vel[:,1] / U_scale
x_data = x_data / X_scale
y_data = y_data / Y_scale

print('Using input data pts: pts: ',x_data, y_data)
print('Using input data pts: vel u: ',data_vel_u)
print('Using input data pts: vel v: ',data_vel_v)
xd= x_data.reshape(-1, 1) #need to reshape to get 2D array
yd= y_data.reshape(-1, 1) #need to reshape to get 2D array
ud= data_vel_u.reshape(-1, 1) #need to reshape to get 2D array
vd= data_vel_v.reshape(-1, 1) #need to reshape to get 2D array


#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train(device,x,y,xb,yb,ub,vb,xd,yd,ud,vd,batchsize,learning_rate,epochs,path,Flag_batch,Diff,rho,Flag_BC_exact,Lambda_BC,nPt,T,xb_inlet,yb_inlet,ub_inlet,vb_inlet )
#tic = time.time()

#elapseTime = toc - tic
#print ("elapse time in serial = ", elapseTime)

 








