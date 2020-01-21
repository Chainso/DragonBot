import torch

class InputFormatter():
    FLOOR = 0
    CEILING = 2044
    MIDDLE_HEIGHT = (CEILING - FLOOR) / 2
    CENTER_FIELD = (0, 0, 0)
    BACK_WALL_LENGTH = 5120
    SIDE_WALL_LENGTH = 4096
    GOAL_HEIGHT = 642.775
    GOAL_CENTER_TO_POST = 892.755

    """
    A class to format game input to feed into the model.
    """
    def __init__(self, team, index, device="cpu"):
        self.team = team
        self.index = index
        self.device = torch.device(device)

    def get_obj_info(self, car):
        """
        Gets the relevant information for a game car.
        """
        physics = car.physics

        loc = physics.location
        rot = physics.rotation
        vel = physics.velocity
        ang_v = physics.angular_velocity

        # Normalize all values, dividing by 1000 for now until max/min
        # velocities are attained
        #loc_x = loc.x / SIDE_WALL_LENGTH
        #loc_y = loc.y / BACK_WALL_LENGTH
        #loc_z = (loc.z - MIDDLE_HEIGHT) / MIDDLE_HEIGHT


        loc_vel = torch.FloatTensor([
            loc.x, loc.y, loc.z,
            vel.x, vel.y, vel.z,
            ang_v.x, ang_v.y, ang_v.z
        ])
        loc_vel = loc_vel / 1000

        rot_info = torch.FloatTensor([
            rot.pitch, rot.yaw, rot.roll,
        ])

        obj_info = torch.cat([loc_vel, rot_info])

        return obj_info

    def transform_packet(self, packet):
        """
        Transforms the packet into a state to feed into the model.
        """
        # It's your car first, then your team, then the enemy team
        my_team_info = []
        enemy_team_info = []

        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            car_info = self.get_obj_info(car)

            if i == self.index:
                boost = torch.FloatTensor([packet.num_boost / 100])
                my_car_info = torch.cat([car_info, boost])
                my_team_info = [my_car_info] + my_team_info
            elif car.team == self.team:
                my_team_info.append(car_info)
            else:
                enemy_team_info.append(car_info)

        ball_info = self.get_obj_info(packet.game_ball)

        car_infos = torch.cat(my_team_info + enemy_team_info)

        all_info = torch.cat(my_team_info + enemy_team_info + [ball_info])

        return all_info.to(self.device)

    def transform_batch(self, packets):
        """
        Transforms a batch of packets.
        """
        return torch.stack([self.transform_packet(packet) for packet in packets])

    @staticmethod
    def state_space():
        """
        Returns the shape of the input state (excluding batch size and sequence
        length).
        """
        return (37,)

    @staticmethod
    def input_space():
        """
        Returns the shape of the formatted input.
        """
        return (1, *InputFormatter.state_space())

class RecurrentInputFormatter(InputFormatter):
    def __init__(self, team, index, device="cpu"):
        super().__init__(team, index, device)

    def transform_packet(self, packet):
        """
        Transforms the packet into a state to feed into the model.
        """
        #### COPYING BECAUSE THE CLASS ISN'T LOADING PROPERLY RIGHT NOW
        # It's your car first, then your team, then the enemy team
        my_team_info = []
        enemy_team_info = []

        for i in range(packet.num_cars):
            car = packet.game_cars[i]
            car_info = self.get_obj_info(car)

            if i == self.index:
                boost = torch.FloatTensor([packet.num_boost / 100])
                my_car_info = torch.cat([car_info, boost])
                my_team_info = [my_car_info] + my_team_info
            elif car.team == self.team:
                my_team_info.append(car_info)
            else:
                enemy_team_info.append(car_info)

        ball_info = self.get_obj_info(packet.game_ball)

        car_infos = torch.cat(my_team_info + enemy_team_info)

        all_info = torch.cat(my_team_info + enemy_team_info + [ball_info])

        return all_info.to(self.device)

    def transform_batch(self, packets):
        """
        Transforms a matrix of (batch size, sequence length, packet shape).
        """
        return torch.stack([torch.stack([self.transform_packet(packet)
                                         for packet in packet_seq])
                            for packet_seq in packets])

    @staticmethod
    def input_space():
        return (1, *InputFormatter.input_space())