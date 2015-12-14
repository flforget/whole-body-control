# coding=utf-8

from IPython import embed
import pinocchio as se3
import scipy
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
from qpoases import PyQProblemB as QProblemB
from qpoases import PySQProblem as SQProblem
from qpoases import PyQProblem as QProblem
from qpoases import PyPrintLevel as PrintLevel
from qpoases import PyOptions as Options

from pinocchio.romeo_wrapper import RomeoWrapper

def display_RH_objective(robot,objective_SE):
    robot.viewer.gui.applyConfiguration("world/pinocchio/caps0",se3.utils.se3ToXYZQUAT(objective_SE))
def display_RH_objective2(robot,objective2_SE):
    robot.viewer.gui.applyConfiguration("world/pinocchio/caps1",se3.utils.se3ToXYZQUAT(objective2_SE))

def display_com_projection(robot,q):
    caps_com = robot.com(q).copy()
    caps_com[2] = 0.0
    com_SE3=se3.SE3(se3.utils.rpyToMatrix(np.matrix([[.0],[.0],[.0]])),caps_com)
    robot.viewer.gui.applyConfiguration("world/pinocchio/capscom",se3.utils.se3ToXYZQUAT(com_SE3))

def errorInSE3( M,Mdes):
    '''
    Compute a 6-dim error vector (6x1 np.maptrix) caracterizing the difference
    between M and Mdes, both element of SE3.
    '''
    error = se3.log(Mdes.inverse()*M)
    return error.vector()

def errorLinkInSE3dyn(linkId,Mdes,v_des,q,v):
    # Get the current configuration of the link
    M = robot.position(q, linkId)
    gMl = se3.SE3.Identity()
    gMl.rotation = M.rotation
    v_frame = robot.velocity(q,v,linkId)
    # Compute error
    error = errorInSE3(M, Mdes);
    v_error = v_frame - gMl.actInv(v_des)

    a_corriolis = robot.acceleration(q,v,0*v,linkId)
    #~ a_corriolis.linear += np.cross(v_frame.angular.T, v_frame.linear.T).T

    #~ a_tot = gMl.actInv(a_corriolis) #a_ref - gMl.actInv(a_corriolis)
    a_tot = a_corriolis
    #~ dJdq = a_tot.vector() *self.dt
    dJdq = a_corriolis.vector()
    return error,v_error.vector() ,dJdq

def null(A, eps=1e-6):#-12
    '''Compute a base of the null space of A.'''
    u, s, vh = np.linalg.svd(A)
    padding = max(0,np.shape(A)[1]-np.shape(s)[0])
    null_mask = np.concatenate(((s <= eps), np.ones((padding,),dtype=bool)),axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


import time
print ("start")

LEFT_HAND_OBJECTIVE = 0

robot = RomeoWrapper("/home/flo/sources/pinocchio/models/romeo.urdf")
robot.initDisplay()
robot.loadDisplayModel("world/pinocchio","pinocchio")

q0 = robot.q0.copy()
v0 = robot.v0.copy()
v0ref = se3.se3.Motion(np.zeros(6))
q = robot.q0.copy()
v = robot.v0.copy()
a = robot.v0.copy()
robot.display(robot.q0)

# simulation timings
dt = 5e-3
final_time = 10.0
N = int(final_time/dt)


Rf0 = robot.Mrf(q0)
Lf0 = robot.Mlf(q0)
com0 = robot.com(q0)

robot.viewer.gui.addSphere("world/pinocchio/capscom",0.02,[0,1,0,1.0])
robot.viewer.gui.addSphere("world/pinocchio/caps0",0.03,[1,0,0,1.0])

robot.viewer.gui.setLightingMode("world/pinocchio/caps0","OFF")
robot.viewer.gui.setLightingMode("world/pinocchio/capscom","OFF")
robot.viewer.gui.setLightingMode("world/floor","OFF")
if(LEFT_HAND_OBJECTIVE):
    robot.viewer.gui.addSphere("world/pinocchio/caps1",0.03,[0,0,1,1.0])
    robot.viewer.gui.setLightingMode("world/pinocchio/caps1","OFF")

# robot.viewer.gui.setLightingMode("world/pinocchio","ON")

options = Options()
options.printLevel = PrintLevel.NONE

## gains
K_min = 100.0

eps_post = 1e-5#e-3
K = 100.0
Kp_post = 200.0
Kd_post = 2*np.sqrt(Kp_post)

Kp_lf = 200.0
Kd_lf = 2*np.sqrt(Kp_lf)

Kp_rf = 200.0
Kd_rf = 2*np.sqrt(Kp_rf)

Kp_rh = 200.0
Kd_rh = 2*np.sqrt(Kp_rh)

Kp_lh = 200.0
Kd_lh = 2*np.sqrt(Kp_lh)

Kp_trunk = 200
Kd_trunk = 2*np.sqrt(Kp_trunk)

grav = 9.81
omega = np.sqrt(grav/robot.com(q0)[2,0])
polym = np.matrix([-0.10,-0.10]).T
polyM = np.matrix([0.10,0.10]).T
ka = (2*omega)/((omega*dt + 2)*dt)
#

objective = np.array([[0.5],[-0.1],[0.6]])
objective_SE3=se3.SE3(se3.utils.rpyToMatrix(np.matrix([[.0],[.0],[.0]])),objective)
display_RH_objective(robot,objective_SE3)
if(LEFT_HAND_OBJECTIVE):
    objective2 = np.array([[0.4],[0.2],[1.2]])
    objective2_SE3=se3.SE3(se3.utils.rpyToMatrix(np.matrix([[.0],[.0],[.0]])),objective2)
    display_RH_objective2(robot,objective2_SE3)

ConstraintsSize = 17
## random initialisation
# q[7:] = robot.q0[7:] + (0.3*(np.matrix(np.random.random(robot.nv-6))-0.5*np.ones(robot.nv-6))).T
robot.display(q)

qp = SQProblem(robot.nv,ConstraintsSize)
qp.setOptions(options)
for i in range(N):
    current_time = i*dt
    # objective = np.array([[0.4],[-0.1+0.1*np.sin(2*np.pi*current_time/final_time)],[1.0-0.4*(current_time/final_time)]])
    # objective_SE3=se3.SE3(se3.utils.rpyToMatrix(np.matrix([[.0],[.0],[.0]])),objective)
    # display_RH_objective(robot,objective_SE3)

    # Update robot geometry
    robot.geometry(q)
    robot.computeJacobians(q)
    robot.mass(q)
    robot.biais(q,v)
    robot.dynamics(q,v,0*v)

    com,comVel,dJdq_com = robot.com(q,v,0*a)
    comAcc_desm = polym - com[:2] - comVel[:2]*(dt+(1/omega))
    comAcc_desM = polyM - com[:2] - comVel[:2]*(dt+(1/omega))

    ## qpOASES resoluation

    ## acceleration minimization
    Alin_min = K_min*np.identity(robot.nv)
    blin_min = np.zeros(robot.nv)
    Alin_min = np.hstack([K_min*np.identity(6),np.zeros((6,robot.nv-6))])
    blin_min = np.zeros(6)

    # posture task
    post_pd = Kp_post*(q - q0)[7:] + Kd_post*(v-v0)[6:]
    Alin_post = np.hstack([np.zeros((robot.nv-6,6)),np.eye(robot.nv-6)])
    blin_post = np.array(post_pd).reshape(robot.nv-6,)

    ## left foot task
    err_lf_pos, err_lf_vel,dJdq_lf = errorLinkInSE3dyn(robot.lf,Lf0,v0ref,q,v)
    err_lf = Kp_lf*err_lf_pos + Kd_lf*err_lf_vel

    ## right foot task
    err_rf_pos, err_rf_vel, dJdq_rf = errorLinkInSE3dyn(robot.rf,Rf0,v0ref,q,v)
    err_rf = Kp_rf*err_rf_pos + Kd_rf*err_rf_vel

    ## right hand task
    err_rh_pos, err_rh_vel, dJdq_rh = errorLinkInSE3dyn(robot.rh,objective_SE3,v0ref,q,v)
    err_rh = Kp_rh*err_rh_pos[:3] + Kd_rh*err_rh_vel[:3]
    Alin_rh = robot.Jrh(q)[:3].copy()
    blin_rh = np.array(err_rh)[:3].reshape(3,)

    ## left hand task
    if(LEFT_HAND_OBJECTIVE):
        err_lh_pos,err_lh_vel, dJdq_lh = errorLinkInSE3dyn(robot.lh,objective2_SE3,v0ref,q,v)
        err_lh = Kp_lh*err_lh_pos[:3] + Kd_lh*err_lh_vel[:3]
        Alin_lh = robot.Jlh(q)[:3].copy()
        blin_lh = np.array(err_lh)[:3].reshape(3,)

    ## trunk task
    Alin_trunk = robot.jacobian(q,robot.index('root'))[3:6].copy()
    MTrunk0=robot.position(robot.q0,robot.index('root'))
    errTrunk,v_errTrunk,dJdqTrunk = errorLinkInSE3dyn(robot.index('root'),MTrunk0,v0ref,q,v)
    blin_trunk = np.array(Kp_trunk*errTrunk[3:6]+Kd_trunk*v_errTrunk[3:6]).reshape(3,)

    ## capture point task
    err_CP_l = (ka*comAcc_desm) - dJdq_com[:2]
    err_CP_u = (ka*comAcc_desM) - dJdq_com[:2]

    ## QP
    Alin = np.vstack([
                        # Alin_min,
                        eps_post*Alin_post,
                        #Alin_rh,
                        # Alin_lh,
                        Alin_trunk,
                                                        ])
    blin = np.hstack([
                        # blin_min,
                        eps_post*blin_post,
                        #blin_rh,
                        # blin_lh,
                        blin_trunk,
                                                        ])
    H = np.dot(Alin.T,Alin)
    g = np.array(np.dot(Alin.T,blin)).reshape(robot.nv,)

    A = np.vstack([
                    robot.Jlf(q).copy(),
                    robot.Jrf(q).copy(),
                    robot.Jrh(q)[:3].copy(),
                    robot.Jcom(q).copy()[:2],
                                    ])

    lbA = np.array(np.vstack([
                                -err_lf-dJdq_lf,
                                -err_rf-dJdq_rf,
                                -err_rh-dJdq_rh[:3],
                                err_CP_l
                                                    ])).reshape(ConstraintsSize,)
    ubA = np.array(np.vstack([
                                -err_lf-dJdq_lf,
                                -err_rf-dJdq_rf,
                                -err_rh-dJdq_rh[:3],
                                err_CP_u
                                                    ])).reshape(ConstraintsSize,)

    lb = -1000*np.ones(robot.nv)
    ub = 1000*np.ones(robot.nv)

    if i==0:
        qp.init(H, g, A, lb, ub, lbA, ubA, np.array([100]))
    else:
        qp.hotstart(H, g, A, lb, ub, lbA, ubA, np.array([100]))
    sol = np.zeros(robot.nv)
    qp.getPrimalSolution(sol)
    a = np.matrix(sol).T

    se3.rnea(robot.model,robot.data,q,v,a)

    tau_joints = robot.data.tau[6:]



    ## configuration integration
    v += np.matrix(a*dt)
    robot.increment(q,v*dt)
    ## Display new configuration
    robot.display(q)
    display_com_projection(robot,q)
    time.sleep(0.005)




'''### manual resolution

## posture task
err_post_pos = Kp_post*(q[7:]-q0[7:])
err_post_vel = Kd_post*(v[6:]-v0[6:])
err_post = eps_post*(err_post_pos + err_post_vel)
J_post = np.hstack([np.zeros([robot.nv-6,6]), np.eye(robot.nv-6) ])

## right hand task
err_rh_pos, err_rh_vel,dJdq_rh = errorLinkInSE3dyn(robot.rh,objective_SE3,v0ref,q,v)
err_rh = Kp_rh*err_rh_pos[:3] + Kd_rh*err_rh_vel[:3]
J_rh = robot.Jrh(q)[:3].copy()

## left foot task
err_lf_pos, err_lf_vel,dJdq_lf = errorLinkInSE3dyn(robot.lf,Lf0,v0ref,q,v)
err_lf = Kp_lf*err_lf_pos + Kd_lf*err_lf_vel
J_lf = robot.Jlf(q).copy()

## right foot task
err_rf_pos, err_rf_vel, dJdq_rf = errorLinkInSE3dyn(robot.rf,Rf0,v0ref,q,v)
err_rf = Kp_rf*err_rf_pos + Kd_rf*err_rf_vel
J_rf = robot.Jrf(q).copy()

## merged resolution
# err = np.vstack([err_post,err_lf,err_rf])
# J = np.vstack([J_post,J_lf,J_rf])
err = np.vstack([err_rh,err_lf,err_rf])
J = np.vstack([J_rh,J_lf,J_rf])
a = np.linalg.pinv(J)*(-err)

## hierarchical resolution
# err1 = np.vstack([err_post,err_lf,err_rf])
# J1 = np.vstack([J_post,J_lf,J_rf])
# err2 = err_rh
# J2 = J_rh
# a = npl.pinv(J2)*(-err2)
# nullProjector1 = null(J2)
# a -= nullProjector1*npl.pinv(J1*nullProjector1)*(-err1 - J1*a)
print a.T*a'''