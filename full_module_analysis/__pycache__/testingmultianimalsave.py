"""

# # # # TESTING # # # # 
metrics = multi_animal_main(path=r'C:\Users\quicken\Code\Ambros_analysis\code_test\soial_behavior',
                  habituation=False,
                  social_inv=True)

individuals = metrics["individuals"]
exp_len = metrics["exp_len"]

save_analysis(
    metrics,
    r'C:\Users\quicken\Code\Ambros_analysis\code_test\soial_behavior\test.joblib'
)

results = load_analysis(r'C:\Users\quicken\Code\Ambros_analysis\code_test\soial_behavior\test.joblib')

print(metrics["thetas"] == results["thetas"])

for ind in metrics["individuals"]:
    print(metrics["face_inv_idx"][ind])

create =  False
if create:
    from create_labelled_video import create_labelled_video_modular
    create_labelled_video_modular(video_path=r'C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4',
                            output_path=r'C:\Users\quicken\Code\social_behavior_analysistest\socialbehavior.mp4',
                            metrics=[
                                ("Face Investigation:", metrics["face_inv"][0]),
                                ("Body Investigation:", metrics["body_inv"][0]),
                                ("Anogenital Investigation:", metrics["anogenital_inv"][0])
                            ],
                            row_gap=20,
                            scale_factor=1.0
    )
create_center = False
if create_center:
    shrinked_coords = shrink_rectangle(ARENA_COORDS, scale=0.6)
    arena_coords = []
    for (x, y) in shrinked_coords:
        arena_coords.append((x, y*-1))
    overlay_metric_at_centers(in_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4",
                            out_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\faceinvoverlay.mp4",
                            centers_xy = metrics["centers_xy"][0],
                            metric = [np.nancumsum(metrics["face_inv"][0]) / np.nancumsum(metrics["mice_presence"][0]) *100],
                            color_mask = metrics["face_inv"][0],
                            unit=["% ind1 faceinv"],
                            marker_radius=8,
                            marker_color_bgr=(245, 66, 194),
                            font_scale=0.8,
                            draw_rect=True,
                            rect_xy = arena_coords,
                            draw_circle=True,
                            circle_xy=metrics["nose_xy"][0],
                            circle_radius=PIXEL_PER_CM*2.0,
                            circle_fill_bgr=(245, 66, 194),
                            circle_outline_bgr=(245, 66, 194)
                            )

thetaplot = False
if thetaplot:
    
    overlay_two_points_line_and_theta_segments(in_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\2025_11_13_12_37_14_mice_omm12prop_females_home_unfamiliar_top1_labeled.mp4",
                                                out_video_path=r"C:\Users\quicken\Code\social_behavior_analysistest\thetas.mp4",
                                                xy1=metrics["fronts_xy"][0],
                                                xy2=metrics["rears_xy"][0],
                                                theta_segments=metrics["theta_dic"][individuals[0]],
                                                trail_len=10
                                                )
    
"""