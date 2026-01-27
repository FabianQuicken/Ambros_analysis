trajectories = [
    {
        "start_frame": 10234,
        "end_frame": 11890,
        "duration_s": 55.3
    }
]


dic = {
        "start_frame": 12234,
        "end_frame": 13890,
        "duration_s": 40.3
    }

trajectories.append(dic)




trajectories2 = [
    {
        "start_frame": 14234,
        "end_frame": 16890,
        "duration_s": 455.3
    },
    {
        "start_frame": 18234,
        "end_frame": 20890,
        "duration_s": 155.3
    }
]

trajectories = trajectories + trajectories2


a = [-1, -5, -10]
print(min(a))