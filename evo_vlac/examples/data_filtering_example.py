from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
#Consistent with the web interface, the value and citic rewards of video input can be evaluated.


#assign local model path
model_path="set to your local model path"

#assign video path and task description
test_video='./videos/pick-bowl-test.mp4'
ref_video='./videos/pick-bowl-ref.mov'#optional
task_description='Put up the bowl and place it back in the white storage box.'

#init model
Critic=GAC_model(tag='critic')
Critic.init_model(model_path=model_path,model_type='internvl2',device_map=f'cuda:0')
Critic.temperature=0.5
Critic.top_k=1
Critic.set_config()
Critic.set_system_prompt()

# transform video
test_video_compressed = os.path.join(os.path.dirname(test_video),"test.mp4")
_,output_fps=compress_video(test_video, test_video_compressed,fps=5)
reference_video_compressed = None
if ref_video:
    reference_video_compressed = os.path.join(os.path.dirname(ref_video),"ref.mp4")
    compress_video(ref_video, reference_video_compressed,fps=5)


# generate Critic results
result_path,value_list,critic_list,done_list = Critic.web_trajectory_critic(
    task_description=task_description,
    main_video_path=test_video_compressed,
    reference_video_path=reference_video_compressed,#if None means no reference video, only use task_description to indicate the task
    batch_num=10,#batch number
    ref_num=6,#image number used in reference video
    think=False,# whether to CoT
    skip=5,#pair-wise step
    rich=False,#whether to output decimal value
    reverse_eval=False,#whether to reverse the evaluation(for VROC evaluation)
    output_path="results",
    fps=float(output_fps),
    frame_skip=True,#whether to skip frames(if false, each frame while be evaluated, cost more time)
    video_output=False
)

value_list=Critic.critic_to_value_simple(critic_list,'mix_f')
voc=Critic.compute_voc(value_list)
nr=Critic.compute_negative_rate(critic_list)
print("=" * 100)
print(">>>>>>>>>DATA diagnose<<<<<<<<<<")
print(" ")
print(f'Negative rate: {nr}')
print(f"VOC: {voc}")
print("The larger the VOC value(-1~+1) and lower Negative rate(0~1), the better the data quality; overly values can directly filter out data, and specific thresholds can be selected based on the specific task.")
print("=" * 50)

print("critic_list:")
print(critic_list)
print("Actions corresponding to steps with a negative Critic can be filtered out to avoid interference with imitation learning by incorrect movements.")
print("=" * 50)