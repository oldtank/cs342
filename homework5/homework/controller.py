import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    target_velocity = 40

    if current_vel > target_velocity:
        action.brake = True

    steer_sign = 0
    if aim_point[0]>0:
        steer_sign = 1
    elif aim_point[0] < 0:
        steer_sign = -1

    steer_magnitude = 0.
    if aim_point[1] == 0:
        steer_magnitude=9999.
    else:
        steer_magnitude=abs(aim_point[0]/aim_point[1])

    if current_vel < target_velocity:
        if steer_magnitude < 10:
            action.acceleration = (target_velocity - current_vel) / target_velocity
    

    action.steer = steer_sign * min(1.25*steer_magnitude, 1)
    if steer_magnitude > 2:
        action.drift = True
    if steer_magnitude > 5:
        action.brake=True
    
    # if aim_point[0] < 0:
    #     if aim_point[0] > -0.25:
    #         action.steer = 4*aim_point[0]
    #     else:
    #         action.steer = -1
    # elif aim_point[0] > 0:
    #     if aim_point[0] > 0.25:
    #         action.steer = 4*aim_point[0]
    #     else:
    #         action.steer = 1
    # if aim_point[0] > 0.25 or aim_point[0] < -0.25:
    #     action.drift = True

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
