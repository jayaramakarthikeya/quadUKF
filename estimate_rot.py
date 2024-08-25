
import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter


def quat2rotv(q):
    qs = q[0]
    qv = q[1:4]
    if np.linalg.norm(qv) == 0:
        v = np.transpose(np.matrix([0, 0, 0]))
    else:
        v = 2 * ((qv / np.linalg.norm(qv)) * math.acos(qs / np.linalg.norm(q)))
    return v


def calibrate(accel,gyro):
    accel[:,[0,1]] = accel[:,[0,1]] * -1
    gyro = np.concatenate((gyro[:,1:],gyro[:,0].reshape(-1,1)),axis=1)
    #accel_bias = np.array([-510.80714286, -500.99428571,  498.72007143])
    accel_bias = np.mean(accel[:20] -np.array([0,0,9.81*1023*50/3300]),axis=0)
    accel_sensitivity = np.array([40.17477143,  40.73725114,  36.33659008])
    scale_factor_accel = 3300/(1023*accel_sensitivity)
    calibrated_accel = (accel - accel_bias)*scale_factor_accel
    #gyro_bias = np.array([369.68571429, 373.57142857, 375.37285714])
    gyro_bias = np.mean(gyro[:20], axis=0)
    gyro_sensitivity = np.array([289.78076315,289.78076315, 289.78076315])
    scale_factor_gyro = 3300/(1023*gyro_sensitivity)
    calibrated_gyro = (gyro - gyro_bias)*scale_factor_gyro*(np.pi/180)
    #gyro_sensitivity = 3.33
    #gyro_scale_factor = 3300/1023/gyro_sensitivity
    #calibrated_gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)
    print("Accelerometer bias: ",accel_bias, " Accelerometer sensitivity: ",accel_sensitivity)

    return calibrated_accel , calibrated_gyro

def generate_sigma_points(sigma,state,n):
    q_k_1 = Quaternion(state[0],vec=state[1:4])
    w_k = state[4:]
    S = np.linalg.cholesky(sigma)
    left = np.sqrt(n) * S
    right = -np.sqrt(n) * S
    W = np.hstack((left, right)).T  
    X = np.zeros((2*n,7))

    for i in range(2*n):
        temp = Quaternion()
        temp.from_axis_angle(W[i,:3])
        X[i,:4] = (q_k_1*temp).q
        X[i,4:] = W[i,3:] + w_k

    return X

def transform_sigma_points(X,n,dt,w_k):
    Y = np.zeros((2*n,7))
    for i in range(2*n):
        alpha_delta = np.linalg.norm(w_k)*dt
        if alpha_delta != 0:
                axis = w_k/alpha_delta
        else:
            axis = np.array([1,0,0])

        scalar = math.cos(alpha_delta/2)
        vec = axis*math.sin(alpha_delta/2)
        q_delta = Quaternion(scalar=scalar,vec=vec)
        q_k = Quaternion(X[i,0],vec=X[i,1:4])
        y_q = q_k*q_delta
        Y[i] = np.concatenate((y_q.q,w_k))


    return Y


def transformed_update(state,Y,n):
    epilson = 0.00001
    mean_q = Quaternion(state[0],vec=state[1:4])
    w_mean = np.mean(Y[:,4:],axis=0)
    w_y = Y[:,4:]
    w_error = w_y - w_mean
    #error_mean = np.zeros((3,))
    for _ in range(1000):
        error = np.zeros((2*n,3))
        for i in range(2*n):
            q_y = Quaternion(scalar=Y[i,0],vec=Y[i,1:4])
            e_i = q_y*mean_q.inv()
            e_i.normalize()
            error_v = quat2rotv(e_i.q)
            if np.round(np.linalg.norm(error_v),10) == 0:
                error[i,:] = np.zeros(3)
            else:
                error[i,:] = (-np.pi + np.mod(np.linalg.norm(error_v) + np.pi, 2 * np.pi)) / np.linalg.norm(error_v) * error_v
        error_mean = np.mean(error,axis=0)
        err_mean_q = Quaternion()
        err_mean_q.from_axis_angle(error_mean)
        mean_q = err_mean_q*mean_q
        mean_q.normalize()
        if np.linalg.norm(error_mean) < epilson:
            state = np.hstack((mean_q.q,w_mean))
            error = np.hstack((error,w_error))
            return state, error
    

def measurement_model(X_measure,n):
    Y_measure = np.zeros((2*n,6))
    for j in range(2*n):
        q_k = Quaternion(scalar=X_measure[j,0],vec=X_measure[j,1:4])
        a = np.array([0,0,1])
        g = Quaternion(scalar=0,vec=a)
        #g.normalize()
        g_dash = q_k.inv()*g*q_k
        #print(g_dash.q)
        w_k = X_measure[j,4:]
        Y_measure[j] = np.concatenate((g_dash.vec(),w_k))
    
    return Y_measure

def find_error(X_measure,mu_k_plus_1_given_k,n):
    E_dash = np.zeros((2*n,6))
    for i in range(2*n):
        q_x_measure = Quaternion(scalar=X_measure[i,0],vec=X_measure[i,1:4])
        q_mu_k_plus_1_given_k = Quaternion(scalar=mu_k_plus_1_given_k[0],vec=mu_k_plus_1_given_k[1:4])
        e_q = q_x_measure*q_mu_k_plus_1_given_k.inv()
        e_q.normalize()
        vec = e_q.axis_angle()
        w_k_x_measure = X_measure[i,4:]
        w_k_mu_k_plus_1_given_k = mu_k_plus_1_given_k[4:]
        w_k_dash = w_k_x_measure - w_k_mu_k_plus_1_given_k
        E_dash[i,:] = np.concatenate((vec,w_k_dash))

    return E_dash

def quat_to_vec_gain(K_I,mu_k_plus_1_given_k):
    K_I_dash_q = Quaternion()
    K_I_dash_q.from_axis_angle(K_I[:3])
    q_mu_k_plus_1_given_k = Quaternion(scalar=mu_k_plus_1_given_k[0],vec=mu_k_plus_1_given_k[1:4])
    state_q = q_mu_k_plus_1_given_k*K_I_dash_q
    state_w = mu_k_plus_1_given_k[4:] + K_I[3:]
    state = np.concatenate((state_q.q,state_w))

    return state

def find_euler_angles(R):
    alpha = np.arctan2(R[1,0],R[0,0])
    beta= np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
    gamma = np.arctan2(R[2,1],R[2,2])

    return alpha , beta , gamma


    
    



# def estimate_rot(data_num=1):
#     #load data
#     imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
#     #vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
#     accel = imu['vals'][0:3,:]
#     gyro = imu['vals'][3:6,:]
#     T = np.shape(imu['ts'])[1]
#     imu_ts = imu['ts'].squeeze()
#     n = 6

#     roll = []
#     pitch = []
#     yaw = []

#     # your code goes here
#     accel = accel.T
#     gyro = gyro.T
#     calibrated_accel , calibrated_gyro = calibrate(accel,gyro)

#     sigma = 0.01*np.identity(n)
#     R = 0.0001*np.identity(n)
#     Q = 0.0001*np.identity(n)
#     print(calibrated_gyro[0])
#     initial_q = np.array([1,0,0,0])
#     w = np.array([1,2,1])
#     state = np.concatenate((initial_q,w))
#     for i in range(T):
#         if i == T - 1:
#             dt = imu_ts[-1] - imu_ts[-2]
#         else:
#             dt = imu_ts[i+1] - imu_ts[i]
        
#         X = generate_sigma_points(sigma,state,n)
#         Y = transform_sigma_points(X,n,dt,calibrated_gyro[i])

#         mu_k_plus1_given_k , error = transformed_update(state,Y,n)
#         sigma_k_plus_1_given_k = np.zeros((6,6))
#         for k in range(2*n):
#             sigma_k_plus_1_given_k += np.outer(error[k,:],error[k,:])
#         sigma_k_plus_1_given_k /= 24
#         sigma_k_plus_1_given_k += R*dt
#         X_measure = generate_sigma_points(sigma_k_plus_1_given_k,mu_k_plus1_given_k,n)
#         Y_measure = measurement_model(X_measure,n)
#         Y_cap = np.mean(Y_measure,axis=0)
#         Y_cap[:3] /= np.linalg.norm(Y_cap[:3])
#         sigma_yy = np.zeros((n,n))
#         sigma_xy = np.zeros((n,n))
        
#         E = (Y_measure - Y_cap)
#         for k in range(2*n):
#             sigma_yy += np.outer(E[k,:],E[k,:])
#             sigma_xy += np.outer(error[k,:],E[k,:])
#         sigma_yy /= 24
#         sigma_xy /= 24
#         sigma_yy += Q*dt
#         Y_imu = np.concatenate((calibrated_accel[i],calibrated_gyro[i]))
#         Y_imu[:3] /= np.linalg.norm(Y_imu[:3])
#         Innovation = Y_imu - Y_cap
#         #sigma_yy += R
#         K = sigma_xy @ np.linalg.inv(sigma_yy)

#         new_mu = np.zeros(6,)
#         new_mu[3:] = mu_k_plus1_given_k[4:]
#         new_mu[:3] = quat2rotv(mu_k_plus1_given_k[:4])

#         mu_gain = new_mu + np.dot(K,Innovation)

#         mu_update = np.zeros(7,)
#         mu_update_q = Quaternion()
#         mu_update_q.from_axis_angle(mu_gain[:3])
#         mu_update[:4] = mu_update_q.q
#         mu_update[4:] = mu_gain[3:]



#         #mu_update = quat_to_vec_gain(K @ Innovation,mu_k_plus1_given_k)
#         sigma = sigma_k_plus_1_given_k - K @ sigma_yy @ K.T
#         state = mu_update
#         q = Quaternion(scalar=state[0],vec=state[1:4])
#         euler_angles = q.euler_angles()
#         roll.append(euler_angles[0])
#         pitch.append(euler_angles[1])
#         yaw.append(euler_angles[2])

#     # roll, pitch, yaw are numpy arrays of length T
#     return roll,pitch,yaw

#estimate_rot(data)



if __name__ == "__main__":
    imu = io.loadmat('imu/imuRaw'+str(3)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(3)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    imu_ts = imu['ts'].squeeze()
    n = 6
    phi = []
    theta = []
    psi = []

    vicon_euler = np.zeros((vicon["rots"].shape[2], 3))
    for i in range(vicon["rots"].shape[2]):
        r = Rot.from_matrix(vicon["rots"][:, :, i])
        vicon_euler[i] = r.as_euler('xyz')

    # your code goes here
    accel = accel.T
    gyro = gyro.T
    calibrated_accel , calibrated_gyro = calibrate(accel,gyro)

    sigma = 0.01*np.identity(n)
    R = 0.0001*np.identity(n)
    Q = 0.0001*np.identity(n)
    print(calibrated_gyro[0])
    initial_q = np.array([1,0,0,0])
    w = np.array([0,0,0])
    predicted_q = initial_q   #initializing
    predicted_w = w
    cov_quat = np.array([0.0, 0.0, 0.0])
    cov_ang = np.array([0.0, 0.0, 0.0])
    state = np.concatenate((initial_q,w))
    for i in tqdm(range(T)):
        if i == T - 1:
            dt = imu_ts[-1] - imu_ts[-2]
        else:
            dt = imu_ts[i+1] - imu_ts[i]
        
        X = generate_sigma_points(sigma,state,n)
        Y = transform_sigma_points(X,n,dt,calibrated_gyro[i])

        mu_k_plus1_given_k , error = transformed_update(state,Y,n)
        sigma_k_plus_1_given_k = np.zeros((6,6))
        for k in range(2*n):
            sigma_k_plus_1_given_k += np.outer(error[k,:],error[k,:])
        sigma_k_plus_1_given_k /= 24
        sigma_k_plus_1_given_k += R*dt
        X_measure = generate_sigma_points(sigma_k_plus_1_given_k,mu_k_plus1_given_k,n)
        Y_measure = measurement_model(X_measure,n)
        Y_cap = np.mean(Y_measure,axis=0)
        Y_cap[:3] /= np.linalg.norm(Y_cap[:3])
        sigma_yy = np.zeros((n,n))
        sigma_xy = np.zeros((n,n))
        
        E = (Y_measure - Y_cap)
        for k in range(2*n):
            sigma_yy += np.outer(E[k,:],E[k,:])
            sigma_xy += np.outer(error[k,:],E[k,:])
        sigma_yy /= 24
        sigma_xy /= 24
        sigma_yy += Q*dt
        Y_imu = np.concatenate((calibrated_accel[i],calibrated_gyro[i]))
        Y_imu[:3] /= np.linalg.norm(Y_imu[:3])
        Innovation = Y_imu - Y_cap
        #sigma_yy += R
        K = sigma_xy @ np.linalg.inv(sigma_yy)

        new_mu = np.zeros(6,)
        new_mu[3:] = mu_k_plus1_given_k[4:]
        new_mu[:3] = quat2rotv(mu_k_plus1_given_k[:4])

        mu_gain = new_mu + np.dot(K,Innovation)

        mu_update = np.zeros(7,)
        mu_update_q = Quaternion()
        mu_update_q.from_axis_angle(mu_gain[:3])
        mu_update[:4] = mu_update_q.q
        mu_update[4:] = mu_gain[3:]



        #mu_update = quat_to_vec_gain(K @ Innovation,mu_k_plus1_given_k)
        sigma = sigma_k_plus_1_given_k - K @ sigma_yy @ K.T
        state = mu_update
        q = Quaternion(scalar=state[0],vec=state[1:4])
        euler_angles = q.euler_angles()
        phi.append(euler_angles[0])
        theta.append(euler_angles[1])
        psi.append(euler_angles[2])
        #state = new_state
        #print(i)

        temp_quat = np.zeros(3)
        temp_ang = np.zeros(3)
        for i in range(3):
            temp_quat[i] = sigma[i,i]
            temp_ang[i] = sigma[i+3,i+3]

        cov_quat = np.vstack((cov_quat, temp_quat))
        cov_ang = np.vstack((cov_ang, temp_ang))
        # print(qt[3:])
        predicted_w = np.vstack((predicted_w, state[4:]))
        predicted_q = np.vstack((predicted_q, q.q))



        #mu_y = np.sum(w*Y,axis=0)
        #sigma_y = np.sum(w*(Y - mu_y) @ (Y - mu_y).T,axis=0)

    x = [*range(T)]
    plt.figure(1)
    plt.subplot(3, 1, 1) 
    plt.plot(vicon_euler[:, 0], 'b', phi, 'g') 

    plt.subplot(3, 1, 2)
    plt.plot(vicon_euler[:, 1], 'b', theta, 'g') 

    plt.subplot(3, 1, 3)
    plt.plot(vicon_euler[:, 2], 'b', psi, 'g') 
    #plt.tight_layout()

    plt.show()
    x = [*range(T+1)]
    fig = plt.figure(2)
    fig.suptitle("Mean of Quaternion")
    plt.plot(x, predicted_q[:,0],label="w")
    plt.plot(x,predicted_q[:,1],label="i")
    plt.plot(x,predicted_q[:,2],label="j")
    plt.plot(x,predicted_q[:,3],label="k")
    plt.xlabel("time")
    plt.ylabel("mean_q")
    plt.legend()

    plt.show()

    fig = plt.figure(3)
    fig.suptitle("Covariance of Quaternion")
    plt.subplot(311)
    plt.plot(cov_quat[:,0])
    plt.ylabel("covariance_q_1")
    plt.subplot(312)
    plt.plot(cov_quat[:,1])
    plt.ylabel("covariance_q_2")
    plt.subplot(313)
    plt.plot(cov_quat[:,2])
    plt.xlabel("time")
    plt.ylabel("covariance_q_3")
    
    plt.show()

    fig = plt.figure(4)
    fig.suptitle("Covariance of Angular Velocity")
    plt.subplot(311)
    plt.plot(cov_ang[:,0])
    plt.ylabel("covariance_w_1")
    plt.subplot(312)
    plt.plot(cov_ang[:,1])
    plt.ylabel("covariance_w_2")
    plt.subplot(313)
    plt.plot(cov_ang[:,2])
    plt.xlabel("time")
    plt.ylabel("covariance_w_3")
    plt.show()

    fig = plt.figure(5)
    fig.suptitle("Mean of Angular Velocity")
    plt.plot(predicted_w[:,0])
    plt.plot(predicted_w[:,1])
    plt.plot(predicted_w[:,2])
    plt.xlabel("time")
    plt.ylabel("mean_w")
    plt.show()
