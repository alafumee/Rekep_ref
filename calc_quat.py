import numpy as np
from scipy.spatial.transform import Rotation as R

q_current = np.array([0.0550, 0.0544, 0.7010, 0.7090])
theta = np.deg2rad(45)  # 旋转 10 度
q_adjust = R.from_euler('x', theta).as_quat()

q_new = R.from_quat(q_adjust) * R.from_quat(q_current)

q_current = np.array([ 0.32212524 ,-0.21799481 , 0.66843535 , 0.63396197])

theta = np.deg2rad(90)

q_adjust = R.from_euler('z', theta).as_quat()

q_new = R.from_quat(q_adjust) * R.from_quat(q_current)

print(q_new.as_quat())

q_current = np.array([ 0.32212524 ,-0.21799481 , 0.66843535 , 0.63396197])

theta = np.deg2rad(20)

q_adjust = R.from_euler('x', theta).as_quat()

q_new = R.from_quat(q_adjust) * R.from_quat(q_current)

print(q_new.as_quat())
