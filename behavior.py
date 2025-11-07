
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle

def follow(ap):
    theta = (ap.theta + np.pi/2 + np.pi) % (2 * np.pi) - np.pi + ap.theta-ap.prev_theta
    rm_x1 = ap.rm_x1 + ap.x-ap.prev_x
    rm_y1 = ap.rm_y1 + ap.y-ap.prev_y
    rm_x2 = ap.rm_x2 + ap.x-ap.prev_x
    rm_y2 = ap.rm_y2 + ap.y-ap.prev_y

    return rm_x1, rm_y1, theta, rm_x2, rm_y2, theta

def docking(dock_id, ap, second=False):
    noise_rate = 0.002
    # N1 = 300
    # n1_rm = 250
    # n1_ap = 250
    # N2 = N1 + 300
    # n2_rm = N1 + 250
    # n2_ap = N1 + 280
    # N3 = N2 + 300
    # n3_rm = N2 + 250
    # n3_ap = N2 + 270
    # N4 = N3 + 300
    # n4_rm = N3 + 280
    # n4_ap = N3 + 250
    # N5 = N4 + 300
    # n5_rm = N4 + 260
    # n5_ap = N4 + 290

    N1 = 15
    n1_rm = 12
    n1_ap = 12
    N2 = N1 + 15
    n2_rm = N1 + 13
    n2_ap = N1 + 14
    N3 = N2 + 15
    n3_rm = N2 + 12
    n3_ap = N2 + 14
    N4 = N3 + 15
    n4_rm = N3 + 14
    n4_ap = N3 + 12
    N5 = N4 + 16
    n5_rm = N4 + 13
    n5_ap = N4 + 15
    # 板子转下来
    if dock_id < N1:
        coef_rm = np.clip((dock_id + 1) / n1_rm, 0, 1)
        coef_ap = np.clip((dock_id + 1) / n1_ap, 0, 1)
        angle = coef_ap * 1.6
        if second:
            x = ap.rm_x1 * coef_rm + (1 - coef_rm) * ap.rm_x2 + (np.random.random() - 0.5)*noise_rate
            y = ap.rm_y1 * coef_rm + (1 - coef_rm) * ap.rm_y2 + (np.random.random() - 0.5)*noise_rate
        else:
            x = ap.rm_x1 + (np.random.random() - 0.5)*noise_rate
            y = ap.rm_y1 + (np.random.random() - 0.5)*noise_rate
        z =  0.63
        height = z - 0.63
        theta = ap.theta + np.pi/2
    # 车上板子
    elif dock_id<N2:
        angle = 1.6
        coef_rm = np.clip((dock_id - N1 + 1)/(n2_rm - N1), 0, 1)
        coef_ap = np.clip((dock_id - N1 + 1)/(n2_ap - N1), 0, 1)
        x = ap.rm_x1 * (1 - coef_rm) + ap.plane_x * coef_rm + (np.random.random() - 0.5) * noise_rate
        y = ap.rm_y1 * (1 - coef_rm) + ap.plane_y * coef_rm + (np.random.random() - 0.5) * noise_rate
        z = 0.63
        height = z - 0.63
        theta = ap.theta + np.pi/2
    # 车和板子一起降下来
    elif dock_id<N3:
        coef_rm = np.clip((dock_id - N2 + 1)/(n3_rm - N2), 0 , 1)
        coef_ap = np.clip((dock_id - N2 + 1)/(n3_ap - N2), 0 , 1)
        angle = 1.6
        x = ap.plane_x + (np.random.random() - 0.5) * noise_rate
        y = ap.plane_y + (np.random.random() - 0.5) * noise_rate
        z = 0.63 * (1 - coef_ap)
        height = z - 0.63
        theta = ap.theta + np.pi/2
    #车出去
    elif dock_id<N4:
        angle = 1.6
        coef_rm = np.clip((dock_id - N3 + 1)/(n4_rm - N3), 0, 1)
        coef_ap = np.clip((dock_id - N3 + 1)/(n4_ap - N3), 0, 1)
        x = ap.plane_x * (1 - coef_rm) + ap.out_x * coef_rm + (np.random.random() - 0.5)*noise_rate
        y = ap.plane_y * (1 - coef_rm) + ap.out_y * coef_rm + (np.random.random() - 0.5)*noise_rate
        z = 0
        height = -0.63
        theta = ap.theta + np.pi/2
    #车转圈，板子转回去的同时收回去
    elif dock_id < N5:
        coef_rm = np.clip((dock_id - N4 + 1)/(n5_rm - N4), 0, 1)
        coef_ap = np.clip((dock_id - N4 + 1)/(n5_ap - N4), 0, 1)
        angle = 1.6 * (1 - coef_ap)
        x = ap.out_x + (np.random.random() - 0.5)*0.003
        y = ap.out_y + (np.random.random() - 0.5)*0.003
        height = 0.63 * (coef_ap - 1)
        z = 0
        theta = ap.theta + np.pi/2 + np.pi * coef_rm
    else:
        return None, True
    dock_id += 1
    return [height, angle, x, y, z, theta, dock_id], False

def inverse_docking(dock_id, ap, rm, second=False):
    noise_rate = 0.002
    # N1 = 300
    # n1_rm = 200
    # n1_ap = 250
    # N2 = N1 + 300
    # n2_rm = N1 + 250
    # n2_ap = N1 + 280
    # N3 = N2 + 300
    # n3_rm = N2 + 250
    # n3_ap = N2 + 270
    # N4 = N3 + 300
    # n4_rm = N3 + 270
    # n4_ap = N3 + 250
    # N5 = N4 + 300
    # n5_rm = N4 + 260
    # n5_ap = N4 + 270


    N1 = 50
    n1_rm = 47
    n1_ap = 12
    N2 = N1 + 15
    n2_rm = N1 + 12
    n2_ap = N1 + 14
    N3 = N2 + 15
    n3_rm = N2 + 12
    n3_ap = N2 + 13
    N4 = N3 + 15
    n4_rm = N3 + 13
    n4_ap = N3 + 12
    N5 = N4 + 15
    n5_rm = N4 + 13
    n5_ap = N4 + 14
    finished = False
    # 板子转下来, 车移到目标位置
    
    if dock_id < N1:
        
        coef_rm = np.clip((dock_id) / n1_rm, 0, 1)
        coef_ap = np.clip((dock_id + 1) / n1_ap, 0, 1)
        angle = coef_ap * 1.6
        z =  0
        height = -0.63*coef_ap
        # if dock_id < n1_rm:
        x = rm.x + (ap.out_x-rm.x)*coef_rm + (np.random.random() - 0.5) * noise_rate
        y = rm.y + (ap.out_y-rm.y)*coef_rm + (np.random.random() - 0.5) * noise_rate
        
        delta_theta = (ap.theta + np.pi/2 -rm.theta + np.pi) % (2 * np.pi) - np.pi
        theta = rm.theta + delta_theta * coef_rm
        # else:
        #     x = rm.x + (np.random.random() - 0.5) * noise_rate
        #     y = rm.y + (np.random.random() - 0.5) * noise_rate

            # theta = rm.theta 

    # 车子移到板子上
    elif dock_id < N2:
        angle = 1.6
        coef_rm = np.clip((dock_id - N1 + 1)/(n2_rm - N1), 0, 1)
        coef_ap = np.clip((dock_id - N1 + 1)/(n2_ap - N1), 0, 1)
        x = ap.plane_x * coef_rm + ap.out_x * (1 - coef_rm) + (np.random.random() - 0.5) * noise_rate
        y = ap.plane_y * coef_rm + ap.out_y * (1 - coef_rm) + (np.random.random() - 0.5) * noise_rate
        z = 0
        height = z - 0.63
        theta = ap.theta + np.pi/2 
    # 车和板子一起升上去
    elif dock_id < N3:
        coef_rm = np.clip((dock_id - N2 + 1)/(n3_rm - N2), 0 , 1)
        coef_ap = np.clip((dock_id - N2 + 1)/(n3_ap - N2), 0 , 1)
        angle = 1.6
        x = ap.plane_x + (np.random.random() - 0.5) * noise_rate
        y = ap.plane_y + (np.random.random() - 0.5) * noise_rate
        z = 0.63 * coef_ap
        height = z - 0.63
        theta = ap.theta + np.pi/2
    #车回去
    elif dock_id < N4:
        angle = 1.6
        coef_rm = np.clip((dock_id - N3 + 1)/(n4_rm - N3), 0, 1)
        coef_ap = np.clip((dock_id - N3 + 1)/(n4_ap - N3), 0, 1)
        x = ap.plane_x * (1 - coef_rm) + ap.rm_x1 * coef_rm + (np.random.random() - 0.5) * noise_rate
        y = ap.plane_y * (1 - coef_rm) + ap.rm_y1 * coef_rm + (np.random.random() - 0.5) * noise_rate
        z = 0.63
        height = 0
        theta = ap.theta + np.pi/2
    #板子转回去
    elif dock_id < N5:
        coef_rm = np.clip((dock_id - N4 + 1)/(n5_rm - N4), 0, 1)
        coef_ap = np.clip((dock_id - N4 + 1)/(n5_ap - N4), 0, 1)
        angle = 1.6 * (1 - coef_ap)
        if second:
            x = ap.rm_x1 * (1 - coef_rm) + ap.rm_x2 * coef_rm + (np.random.random() - 0.5) * noise_rate
            y = ap.rm_y1 * (1 - coef_rm) + ap.rm_y2 * coef_rm + (np.random.random() - 0.5) * noise_rate
        else:
            x = ap.rm_x1 + (np.random.random() - 0.5) * noise_rate
            y = ap.rm_y1 + (np.random.random() - 0.5) * noise_rate
        height = 0
        z = 0.63
        theta = ap.theta + np.pi/2
    else:
        return None, True
    dock_id += 1
    return [height, angle, x, y, z, theta, dock_id], False

def smooth(x, y, ratio):
    n = len(x)
    m = int(n*ratio)
    

    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)  # Euclidean distances
    cumulative_distance = np.insert(np.cumsum(distances), 0, 0)  # Cumulative distances
    from scipy.interpolate import interp1d
    even_distance = cumulative_distance[-1]*2/(n+m-1)
    even_distances = np.zeros(n)
    for i in range(n):
        if i<m:
            even_distances[i] = i*even_distance
        else:
            even_distances[i] = even_distances[i-1]+(n-i-1)/(n-m)*even_distance
    # Step 3: Interpolate to find new (x, y)
    interp_x = interp1d(cumulative_distance, x, kind='linear')
    interp_y = interp1d(cumulative_distance, y, kind='linear')

    x_smooth = interp_x(even_distances)
    y_smooth = interp_y(even_distances)
    return x_smooth, y_smooth

    


def traj4apollo():

    time_step = 0.01*15
    x_smooth = []
    y_smooth = []
    pick_up_coords = []
    plt.figure(figsize=(8, 6))
    def savetrj(x,y,phase):
        x = np.hstack(x)
        y = np.hstack(y)
        x, y = smooth(x, y, 0.8)
        trajectory_data = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
        with open('Demo/desired_trajectory/apollo_phase_'+str(phase)+'.pkl', "wb") as file:
            pickle.dump(trajectory_data, file)
        plt.scatter(x, y, s=1)  # Interpolated curve
        return [], []
    ## 1st curve
    start = np.array([-80, 5])
    # end = np.array([-55, -15])
    end = np.array([-50, -15])
    pick_up_coords.append(end)
    N = int(np.linalg.norm(end-start)/time_step)
    x_temp = np.linspace(start[0], end[0], N)
    y_temp = (start[1]-end[1])/2*np.cos(np.pi/(end[0]-start[0])*(x_temp-start[0]))+0.5*(start[1]+end[1])
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)
    x_smooth, y_smooth = savetrj(x_smooth, y_smooth, 0)


    ## 2nd curve
    start = end.copy()
    end = np.array([-15, 0])
    N = int(np.linalg.norm(end-start)/time_step)
    x_temp = np.linspace(start[0], end[0], N)
    y_temp = -(end[1]-start[1]) * np.sqrt(1-(x_temp - start[0])**2/(end[0]-start[0])**2)
    
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)

    ## 3nd curve
    start = end.copy()
    # pickup_point = np.array([-30, 15])
    pickup_point = np.array([-35, 15])
    end[0] = 2*pickup_point[0] - start[0]
    end[1] = start[1]
    N = int(np.linalg.norm(pickup_point-start)/time_step)*2
    x_temp = np.linspace(start[0], 2*pickup_point[0]-start[0], N)
    y_temp = (pickup_point[1]-start[1]) * np.sqrt(1-(x_temp - pickup_point[0])**2/(pickup_point[0]-start[0])**2)
    pick_up_coords.append(pickup_point)
    x_smooth.append(x_temp[:int(len(x_temp)/2)])
    y_smooth.append(y_temp[:int(len(x_temp)/2)])

    x_smooth, y_smooth = savetrj(x_smooth, y_smooth, 1)
    x_smooth.append(x_temp[int(len(x_temp)/2):])
    y_smooth.append(y_temp[int(len(x_temp)/2):])
    
    ## 4th curve
    start = end.copy()
    end = np.array([-30, -15])
    N = int(np.linalg.norm(end-start)/time_step)
    x_temp = np.linspace(start[0], end[0], N)
    y_temp = (end[1]-start[1]) * np.sqrt(1-(x_temp - end[0])**2/(end[0]-start[0])**2)
    pick_up_coords.append(end)
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)
    x_smooth, y_smooth = savetrj(x_smooth, y_smooth, 2)

    ## 5th curve
    start = end.copy()
    print(start)
    end = np.array([5, 35])
    N = int(np.linalg.norm(end-start)/time_step)
    x_temp = np.linspace(start[0], end[0], N)
    y_temp = -(end[1]-start[1]) * np.sqrt(1-(x_temp - start[0])**2/(end[0]-start[0])**2)+end[1]
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)
    

    ## 6nd curve
    start = end.copy()
    pickup_point = start.copy()
    pickup_point[0] = -10
    upper_point = np.array([(pickup_point[0]+start[0])/2, start[1]+10])
    pick_up_coords.append(pickup_point)
    N = int(np.linalg.norm(upper_point-start)/time_step)*2
    x_temp = np.linspace(start[0], pickup_point[0], N)
    y_temp = (upper_point[1]-start[1]) * np.sqrt(1-(x_temp - upper_point[0])**2/(upper_point[0]-start[0])**2)+start[1]
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)
    x_smooth, y_smooth = savetrj(x_smooth, y_smooth, 3)

    

    
    

    
    
    
    
    pick_up_coords = np.array(pick_up_coords)
    with open("Demo/desired_trajectory/desired_pickup.pkl", "wb") as file:
        pickle.dump(pick_up_coords, file)

    # x_smooth, y_smooth = smooth(x_smooth, y_smooth, 0.8)
    # trajectory_data = np.hstack((x_smooth.reshape(-1,1), y_smooth.reshape(-1,1)))
    # with open("Demo/desired_apollo.pkl", "wb") as file:
    #     pickle.dump(trajectory_data, file)
    # Plotting
    
    plt.scatter(pick_up_coords[:,0], pick_up_coords[:,1], color='red', label="Pickup location") 
    # plt.scatter(x_smooth, y_smooth, s=1, label="Smooth Curve", color='blue')  # Interpolated curve
    plt.title("Smooth 2D Curve")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect(1)
    plt.grid()
    plt.savefig('test.png')

def traj4rm():
    with open('Demo/desired_pickup.pkl','rb') as f:
        desired_pickup = np.array(pickle.load(f))
    x_smooth = []
    y_smooth = []
    time_step = 0.07
    start1_x = desired_pickup[0, 0]
    start1_y = desired_pickup[0, 1] - 1

    start2_x = desired_pickup[1, 0]
    start2_y = desired_pickup[1, 1] + 1

    N = int(np.sqrt((x1-start1_x)**2+(y1-start1_y))/time_step)
    x1 = -45
    y1 = -35
    x_temp = np.linspace(start1_x, x1, N)
    y_temp = (-y1+start1_y)/(-x1+start1_x)**3*(x_temp-start1_x)**3+start1_y
    x_smooth.append(x_temp)
    y_smooth.append(y_temp)

    N = (-5-x1)/time_step
    x_temp = np.linspace(x1, -5, N)
    y_temp = y1*np.ones(N)

    x_smooth.append(x_temp)
    y_smooth.append(y_temp)

    x2 = -20
    y2 = y1
    N = (-5 - x1)/time_step
    x_temp = np.linspace(x1, -5, N)


    end1_x = desired_pickup[2, 0]
    end1_y = desired_pickup[2, 1] - 1

    end2_x = desired_pickup[3, 0] - 1
    end2_y = desired_pickup[3, 1]


if __name__ == '__main__':
    traj4apollo()