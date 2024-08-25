from scipy import io
import numpy as np
import matplotlib.pyplot as plt

def combine_mats(data_nums):
    accel = []
    gyro = []
    vicon_rots = []
    for data_num in data_nums:
        imu = io.loadmat("imu/imuRaw"+str(data_num))
        accel_ = imu["vals"][0:3,:].astype(np.float64)
        gyro_ = imu["vals"][3:6,:].astype(np.float64)
        vicon = io.loadmat("vicon/viconRot"+str(data_num)+".mat")
        vicon_rots_ = vicon['rots'].astype(np.float64)
        if vicon_rots_.shape[0] < accel_.shape[0]:
            accel_ = accel_[:vicon_rots.shape[0],:]
            gyro_ = gyro_[:vicon_rots.shape[0],:]
        else:
            vicon_rots_ = vicon_rots_[:accel_.shape[0],:]
        
        accel.append(accel_.T)
        gyro.append(gyro_.T)
        vicon_rots.append(vicon_rots_.transpose((2,0,1)))

    accel = np.concatenate(accel,axis=0)
    accel[:,[0,1]] = accel[:,[0,1]] * -1
    gyro = np.concatenate(gyro,axis=0)
    vicon_rots = np.concatenate(vicon_rots,axis=0)
    accel = accel[:vicon_rots.shape[0],:]
    gyro = gyro[:vicon_rots.shape[0],:]

    return accel , gyro , vicon_rots , accel.shape[0]

def find_euler_angles(R):
    alpha = np.arctan2(R[1,0],R[0,0])
    beta= np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
    gamma = np.arctan2(R[2,1],R[2,2])

    return alpha , beta , gamma

def find_bias(accel,gyro):
    accel_bias = np.mean(accel - np.array([0,0,9.81*1023*34/3300]),axis=0)
    gyro_bias = np.mean(gyro,axis=0)
    return accel_bias , gyro_bias

def solve_ls_accel(_accel,vicon_rots):
    W_A = np.array([0,0,-9.81]).reshape(-1,1)
    Y = (1023/3300)*(np.linalg.inv(vicon_rots) @ W_A).squeeze()  
    accel_K = np.linalg.lstsq(_accel,Y,rcond=None)[0]

    return accel_K


if __name__ == "__main__":

    data_nums = [1,2,3]
    accel , gyro, vicon_rots , N = combine_mats(data_nums)
    alpha = []
    beta = []
    gamma = []
    for i in range(N):
        a , b, g = find_euler_angles(vicon_rots[i,:,:])
        alpha.append(a)
        beta.append(b)
        gamma.append(g)

    x = [*range(N)]
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1) 
    plt.plot(x, alpha, 'r-')

    plt.subplot(3, 1, 2)
    plt.plot(x,beta, 'g-')

    plt.subplot(3, 1, 3)
    plt.plot(x,gamma,'b-')
    plt.tight_layout()

    plt.show()

    gyro = np.concatenate((gyro[:,1:],gyro[:,0].reshape(-1,1)),axis=1)
    accel_bias , gyro_bias = find_bias(accel,gyro)
    accel_bias = np.array([-510.80714286, -500.99428571,  498.72007143])
    _accel = accel - accel_bias

    _gyro = gyro - gyro_bias
    accel_K = solve_ls_accel(_accel,vicon_rots)
    #accel_sensitivity = np.array([1/accel_K[0,0],-1/accel_K[1,1],1/accel_K[2,2]])

    accel_sensitivity = np.array([43.17477143,  43.73725114,  36.33659008])

    scale_factor = 3300/(1023*accel_sensitivity)
    calibrated_a = (_accel)*scale_factor


    beta_accel = np.arctan2(calibrated_a[:,0],np.sqrt(calibrated_a[:,1]**2 + calibrated_a[:,2]**2))
    gamma_accel = np.arctan2(calibrated_a[:,1],np.sqrt(calibrated_a[:,0]**2+calibrated_a[:,2]**2))
    x = [*range(N)]
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x,beta_accel,)
    plt.plot(x,beta)

    plt.subplot(2, 1, 2)
    plt.plot(x,gamma_accel)
    plt.plot(x,gamma)
    plt.tight_layout()

    plt.show()


    true_angular_vel = []
    for i in range(1,N-1):
        omega = np.linalg.inv(vicon_rots[i]) @ (vicon_rots[i+1] - vicon_rots[i] / 0.009963035583496094)
        #print(omega)
        w = np.zeros((3,1))
        w[0] = omega[2,1] - omega[1,2] /2
        w[1] = omega[0,2] - omega[2,0] / 2
        w[2] = omega[1,0] - omega[0,1]/2
        true_angular_vel.append(w.T)
    true_angular_vel = np.concatenate(true_angular_vel,axis=0)

    gyro_corrected = np.concatenate((gyro[:,1:],gyro[:,0].reshape(-1,1)),axis=1)
    gyro_bias = find_bias(gyro_corrected[:10],np.array([0,0,0]))
    _gyro = gyro_corrected - gyro_bias
    gyro_K = np.linalg.lstsq(_gyro[1:-1],(1023*np.pi/3300*180)*true_angular_vel,rcond=None)[0]
    gyro_sensitivity = np.array([1/gyro_K[0,0],1/gyro_K[1,1],1/gyro_K[2,2]])

    gyro_sensitivity = np.array([289.12076315,289.3807, 289.78076315])
    scale_factor_gyro = (3300/(1023*gyro_sensitivity))*(np.pi/180)
    calibrated_gyro = (_gyro)*scale_factor_gyro

    x = [*range(N)]
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1) 
    plt.plot(x,calibrated_gyro[:,0],'r-')
    plt.plot(x[1:-1],true_angular_vel[:,0],'b-')

    plt.subplot(3, 1, 2)
    plt.plot(x,calibrated_gyro[:,1],'r-')
    plt.plot(x[1:-1],true_angular_vel[:,1],'b-')

    plt.subplot(3, 1, 3)
    plt.plot(x,calibrated_gyro[:,2],'r-')
    plt.plot(x[1:-1],true_angular_vel[:,2],'b-')
    plt.tight_layout()

    plt.show()
    plt.show()
