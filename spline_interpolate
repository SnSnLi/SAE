import numpy as np
from scipy.interpolate import CubicSpline

def spline_interpolate(points, num_points=100):
    x = np.linspace(0, len(points)-1, len(points))
    x_new = np.linspace(0, len(points)-1, num_points)
    cs = CubicSpline(x, points)
    return cs(x_new)

def spline_interpolate_2d(points, num_points=100):
    x = points[:,0]
    y = points[:,1]
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    
    return np.column_stack((x_new, y_new))

def spline_interpolate_3d(points, num_points=100):
    x = points[:,0]
    y = points[:,1] 
    z = points[:,2]
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    cs_z = CubicSpline(t, z)
    
    x_new = cs_x(t_new)
    y_new = cs_y(t_new)
    z_new = cs_z(t_new)
    
    return np.column_stack((x_new, y_new, z_new))

if __name__ == '__main__':
    # 1D
    points_1d = np.array([1, 5, 2, 8, 3])
    interpolated_1d = spline_interpolate(points_1d)
    
    
