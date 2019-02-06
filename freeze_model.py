from tensorflow.python.tools import freeze_graph

# Freeze the graph
save_path="C:/Users/nermi/PycharmProjects/DronePathTracking/model_files/" #directory to model files
MODEL_NAME = 'my_model' #name of the model optional
input_graph_path = save_path+'save_graph.pbtxt'#complete path to the input graph
checkpoint_path = save_path+'model.ckpt' #complete path to the model's checkpoint file
input_saver_def_path = ""
input_binary = False
output_node_names = "X, Y" #output node's name. Should match to that mentioned in your code
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = save_path+'frozen_'+MODEL_NAME+'.pb' # the name of .pb file you would like to give
clear_devices = True


def freeze():
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_frozen_graph_name, clear_devices, "")


freeze()