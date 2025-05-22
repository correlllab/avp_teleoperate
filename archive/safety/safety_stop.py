from unitree_dds_wrapper.idl import unitree_go, unitree_hg
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription
import numpy as np


def safety_stop():
    # Parameters
    joint_index = 13 # arm index 13-26
    saftey_tau = 0.0 # tau to set when safety stop is triggered
    limit = 0.0 # belive its in Nm but not 100% sure

    try:
        pub = Publisher(unitree_hg.msg.dds_.LowCmd_, "rt/lowcmd",None, 200000)
        sub = Subscription(unitree_hg.msg.dds_.LowState_, "rt/lowstate")
        message = unitree_hg.msg.dds_.LowCmd_()
        
        # Initialize message fields as needed
      
        sub.wait_for_connection()
        print("safety stop initialized")
        
        
    except Exception as e:
        print(f"Error initializing safety stop: {e}")
        return
   
    while True:
        state_msg = sub.msg.motor_state[joint_index].tau_est
        print(f"{state_msg=}, {joint_index=}")
        joint_index += 1
        if state_msg > np.abs(limit): 
            while True:
                print(f"Safety stop triggered: {state_msg=}")
                message.motor_cmd[13].tau_ff = saftey_tau
                message.motor_cmd[14].tau_ff = saftey_tau
                message.motor_cmd[15].tau_ff = saftey_tau
                message.motor_cmd[16].tau_ff = saftey_tau
                message.motor_cmd[17].tau_ff = saftey_tau
                message.motor_cmd[18].tau_ff = saftey_tau
                message.motor_cmd[19].tau_ff = saftey_tau
                message.motor_cmd[20].tau_ff = saftey_tau
                message.motor_cmd[21].tau_ff = saftey_tau
                message.motor_cmd[22].tau_ff = saftey_tau
                message.motor_cmd[23].tau_ff = saftey_tau
                message.motor_cmd[24].tau_ff = saftey_tau
                message.motor_cmd[25].tau_ff = saftey_tau
                message.motor_cmd[26].tau_ff = saftey_tau
                pub.msg = message
                pub.write()
       
        if joint_index > 26:
            joint_index = 13

if __name__ == "__main__":
    safety_stop()
