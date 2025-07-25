from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
#Example code for inputting images and evaluating pair-wise

#assign local model path
model_path="set to your local model path"

#Input n images, output a critic_list of length n-1 and a value_list of length n. The critic evaluates the results of adjacent images (i, i+1); if i+1 is closer to accomplishing the task than i, the evaluation result is positive; otherwise, it is negative. It can evaluate the action rewards between any pair-wise images. The value_list is calculated based on the critic.
test_images=['./images/test/595-6-565-0.jpg','./images/test/595-44-565-0.jpg','./images/test/595-134-565-0.jpg','./images/test/595-139-565-0.jpg','./images/test/595-292-565-0.jpg','./images/test/595-354-565-0.jpg']
#（optional）Input up to 11 images as reference trajectories for tasks, significantly improving adaptation to new tasks and environments.
ref_images=['./images/ref/599-0-521-0.jpg','./images/ref/599-100-521-0.jpg','./images/ref/599-200-521-0.jpg','./images/ref/599-300-521-0.jpg','./images/ref/599-400-521-0.jpg','./images/ref/599-457-521-0.jpg']
task_description='Scoop the rice into the rice cooker.'

#init model
Critic=GAC_model(tag='critic')
Critic.init_model(model_path=model_path,model_type='internvl2',device_map=f'cuda:0')
Critic.temperature=0.5
Critic.top_k=1
Critic.set_config()
Critic.set_system_prompt()

# generate Critic results
critic_list, value_list=Critic.get_trajectory_critic(
    task=task_description,
    image_list=test_images,
    ref_image_list=ref_images,
    batch_num=10,#max batch number when generating critic
    ref_num=len(ref_images),#image number used in ref_images
    rich=False,#whether to output decimal value
    reverse_eval=False,#whether to reverse the evaluation(for VROC evaluation)
)


print("=" * 100)
print(">>>>>>>>>Critic results<<<<<<<<<<")
print(" ")

print("value_list:")
print(value_list)
print("=" * 50)

print("critic_list:")
print(critic_list)
print("=" * 50)