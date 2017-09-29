import scipy.io as sio
import numpy as np
import math

matfile = 'C:\Users\Lenovo\Desktop\CV\project1\CV1_data.mat'
data = sio.loadmat(matfile)

Nx = data['Nx']
Ny = data['Ny']
Nz = data['Nz']
X = data['X']
Y = data['Y']
Z = data['Z']
N = np.zeros((7710,3),dtype=float)
N[:,0] = Nx[:,0]
N[:,1] = Ny[:,0]
N[:,2] = Nz[:,0]
xyz = np.zeros((7710,4),dtype=float)
xyz[:,0] = X[:,0]
xyz[:,1] = Y[:,0]
xyz[:,2] = Z[:,0]
xyz[:,3] = 1
#%% raw data
T = np.array([[-14.0], [-71.0], [1000.0]])
R1 = np.array([[1.0,0,0], [0,1.0,0], [0,0,1.0]])
R2 = np.array([[0.9848,0,0.1736], [0,1.0,0], [-0.1736,0,0.9848]])
f1 = 40.0
f2 = 30.0
sx = sy = 8.0
c0 = r0 = 50.0
a = 0
b = 1.0
d = 33.0
ru = 1.0
L1 = np.array([[0,0,-1.0]])
L2 = np.array([[0.5774,-0.5774,-0.5774]])
length = 7710
#%% writing function
def write_pgm_image(image_name, img, header):
    
    fhand = open(image_name, 'wb')
    res = 'P5\n# Project 1\n%d %d\n%d\n' % (header[0], header[1], header[2])
    
    fhand.write(bytearray(res, 'utf-8'))
    img_data = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            img_data.append(int(img[i][j]))
    
    fhand.write(bytearray(img_data))
    fhand.close()
#%% 
def I_data(f, L):
    I = b*ru*np.pi*(d/f)**2*math.cos(a)**4*np.dot(L,N.transpose())/4
    return (abs(I)+I)*255/2
#%%
def dic(data, zc):
    dict_c_r = {}
    for i in range(7710):
        c = int(round(data[i][0]))
        r = int(round(data[i][1]))
        if (c, r) in dict_c_r.keys():
            if zc[i] < dict_c_r[(c,r)][1]:
                dict_c_r[(c,r)] = (int(data[i][2]), zc[i])
            else:
                pass
        else:
            dict_c_r[(c,r)] = (int(data[i][2]), zc[i])
    return dict_c_r
#%%
def full(f, L, R, image_name):
    c_r = np.dot(np.dot(np.array([[f*sx,0,c0],[0,f*sy,r0],[0,0,1.0]]),np.concatenate((R,T),axis=1)),xyz.transpose())
    c_r[0,:] = c_r[0,:]/c_r[2,:]
    c_r[1,:] = c_r[1,:]/c_r[2,:]
    Idata = I_data(f,L)
    zc = np.dot(R[2], xyz[:,0:3].transpose())
    c_r = np.concatenate((c_r[0:2,:],Idata),axis=0)
    dict_c_r = dic(c_r.transpose(), zc)
    maps = np.zeros((100,100),dtype=int)
    for i in dict_c_r.keys():
        maps[i[1]][i[0]] = dict_c_r[i][0]
    '''for i in range(7710):
        maps[int(round(c_r[:,i][1])),int(round(c_r[:,i][0]))] = int(c_r[:,i][2])'''
        
    header = [100,100,255]
    write_pgm_image(image_name, maps, header)
#%%
def weak(f, L, R, image_name):
    Idata = I_data(f,L)
    zc_bar = np.average(np.dot(R[2,:], xyz[:,0:3].transpose()))+float(T[2])
    R = np.concatenate((R[0:2,:],np.array([[0,0,0]])),axis=0)
    T_2 = np.concatenate((T[0:2,:],np.array([[1]])),axis=0)
    R_T = np.concatenate((R,T_2),axis=1)
    c_r = np.dot(np.dot(np.array([[f*sx,0,c0*zc_bar],[0,f*sy,r0*zc_bar],[0,0,zc_bar]]),R_T),xyz.transpose())
    c_r[0,:] = c_r[0,:] / zc_bar
    c_r[1,:] = c_r[1,:] / zc_bar
    c_r = np.concatenate((c_r[0:2,:],Idata),axis=0)
    maps = np.zeros((100,100),dtype=int)
    
    for i in range(7710):
        maps[int(round(c_r[:,i][1])),int(round(c_r[:,i][0]))] = int(c_r[:,i][2])
    
    header = [100,100,255]
    write_pgm_image(image_name, maps, header)
#%%
def ortho(f, L, R, image_name):
    Idata = I_data(f,L)
    R = np.concatenate((R[0:2,:],np.array([[0,0,0]])),axis=0)
    T_2 = np.concatenate((T[0:2,:],np.array([[1]])),axis=0)
    R_T = np.concatenate((R,T_2),axis=1)
    c_r = np.dot(np.dot(np.array([[sx,0,c0],[0,sy,r0],[0,0,1]]),R_T),xyz.transpose())
    c_r[0,:] = (c_r[0,:] + 1000) / 20
    c_r[1,:] = (c_r[1,:] + 1000) / 20
    c_r = np.concatenate((c_r[0:2,:],Idata),axis=0)
    maps = np.zeros((100,100),dtype=int)
    
    for i in range(7710):
        maps[int(round(c_r[:,i][1])),int(round(c_r[:,i][0]))] = int(c_r[:,i][2])
    header = [100, 100, 255]
    write_pgm_image(image_name, maps, header)
#%%  
ortho(f1,L1,R1,'ortho.pgm')
weak(f1,L1,R1,'weak.pgm')
full(f1,L1,R1,'f1_L1_R1.pgm')
full(f1,L2,R1,'f1_L2_R1.pgm')
full(f2,L1,R1,'f2_L1_R1.pgm')
full(f2,L2,R1,'f2_L2_R1.pgm')
full(f1,L1,R2,'f1_L1_R2.pgm')
full(f1,L2,R2,'f1_L2_R2.pgm')
full(f2,L1,R2,'f2_L1_R2.pgm')
full(f2,L2,R2,'f2_L2_R2.pgm')