from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
#Consistent with the web interface, the value and citic rewards of video input can be evaluated.


#assign local model path
model_path="set to your local model path"
model_path='/cpfs04/user/zhangqi/zhangqi/data/model_garden/0619_intern8b_v7-1-part15-19-resize-decay/v1-20250624-140732/checkpoint-3125'

#assign video path and task description
test_video='./videos/pick-bowl-test.mp4'
ref_video='./videos/pick-bowl-ref.mov'
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
    done_flag=False,#whether to out put done value
    in_context_done=False,#whether use reference video to generate done value
    done_threshold=0.9,#done threshold
    video_output=True#whether to output video
)


print("=" * 100)
print(">>>>>>>>>Critic results<<<<<<<<<<")
print(" ")

print(f"result path: {result_path}")
print(f"task description: {task_description}")
print("=" * 50)

print("value_list:")
print(value_list)
print("=" * 50)

print("critic_list:")
print(critic_list)
print("=" * 50)

print("done_list:")
print(done_list)
print("=" * 100)