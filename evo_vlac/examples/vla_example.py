from evo_vlac import GAC_model
from evo_vlac.utils.video_tool import compress_video
import os
#Example code for realizing VLA inference with open tasks and multi-perspective generalization capability, inputs 1-3 images and eef position, outputs delta eef actions, most friendly to songling (pika) robotic arms.
#This is a preview version, fine-tuning code for different robotic arms will be released together with the paper.

#assign local model path
model_path="set to your local model path"

view_images=["./images/test/595-44-565-0.jpg","./images/test/595-44-565-2.jpg"]
eef_position=[
    -18151,
    11685,
    418833,
    -124631,
    65461,
    -133783,
    27510
]#The unit of xyz is 0.001mm, and the unit of rpy is 0.001 degrees
task_description='Scoop the rice into the rice cooker.'


history=False#whether to use history chat
complete_requests_list=None#chat history

#init model
Policy=GAC_model(tag='Policy')
Policy.init_model(model_path=model_path,model_type='internvl2',device_map=f'cuda:0')
Policy.temperature=0.5
Policy.top_k=1
Policy.set_config()
Policy.set_system_prompt()

query=Policy.get_action_prompt(task=task_description,view_num=len(view_images),position_output=False,simple=False,state=Policy.format_state(eef_position,gripper_format=False),think=False)
infer_requests=Policy.get_infer_requests(prompt=query,images=view_images)
if history:
    if complete_requests_list:
        complete_requests_list[0].images.extend(infer_requests[0].images)
        complete_requests_list[0].messages.append(infer_requests[0].messages[1])
        if len(complete_requests_list[0].images)>history_image_num:
            complete_requests_list[0].images=complete_requests_list[0].images[len(infer_requests[0].images):]
            complete_requests_list[0].messages=complete_requests_list[0].messages[:1]+complete_requests_list[0].messages[3:]
        infer_requests=complete_requests_list
response_list,infer_time=Policy.chat(infer_requests)
answers_list,complete_requests_list=Policy.results_format(response_list,infer_requests,rich=True)


print("=" * 100)
print(">>>>>>>>>VLA results<<<<<<<<<<")
print(" ")
print(f'action:{answers_list}')