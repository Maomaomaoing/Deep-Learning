import onnx
from onnx import helper, numpy_helper, shape_inference, version_converter
from onnx import TensorProto

"""
## Operator. Have input, output, attribute
model.graph.node

## Tensor of defined value (weight, layer parameter)
model.graph.initializer

## model input (sometimes need include initializer)
model.graph.input

## 
model.graph.output
"""

## get onnx version
def get_version(model):
    ## == netron model properties format
    ## ex. ONNX v5
    print("ir version:", model.ir_version)
    ## == netron model properties imports
    ## ex. ai.onnx v12
    print("opset version:", model.opset_import)

## initializer
def print_all_initializer(graph):
    ## initializer: have name(weight name or id), data(weight or shape)
    ## conv -> data: weight & bias
    ## reshape -> data: shape
    weights = graph.initializer
    for i,w in enumerate(weights):
        weight = numpy_helper.to_array(w)
        print(w.name, weight.shape)
        # print(weight)
        # print()

def print_initializer(graph, data_id):
    ## initializer: have name(weight name or id), data(weight or shape)
    ## conv -> data: weight & bias
    ## reshape -> data: shape
    weight = graph.initializer
    for i,w in enumerate(weight):
        weight = numpy_helper.to_array(w)
        print(w.name, weight.shape)
        print(weight)
        print()

def add_initializer(graph):
    scales = helper.make_tensor("Resize_scales", TensorProto.FLOAT, [4], [1,1,1,1])
    roi = helper.make_tensor("Resize_roi", TensorProto.FLOAT, [8], [0,0,0,0,0,0,0,0])
    graph.initializer.append(scales)
    graph.initializer.append(roi)

## output
def print_output(graph):
    print(graph.output)

def insert_output(graph, feat_name_shape):
    ## feat_name_shape: layer out feat name and shape
    for i, n in enumerate(graph.node):
            if n.output[0] in feat_name_shape:
                print(n.output[0])
                tmp_out = helper.make_tensor_value_info(n.output[0], TensorProto.FLOAT, feat_name_shape[n.output[0]])
                print(tmp_out)
                graph.output.extend([tmp_out])

## input
def print_input(graph):
    print(graph.input)

def insert_input(graph):
    graph.input.remove(graph.input[0])
    tmp_in = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, (1, 3, 416, 416))
    graph.input.insert(0, tmp_in)
    print(graph.input)

def add_input_shape(graph):
    ## add input shape
    graph.input.remove(graph.input[0])
    tmp_in = helper.make_tensor_value_info("input_1", TensorProto.FLOAT, (1, 3, 416, 416))
    graph.input.insert(0, tmp_in)
    # print(graph.input)

def add_initializer_to_input(graph):
    ## add initializer to input
    inits = []
    for init in graph.initializer:
        weight = numpy_helper.to_array(init)
        # print(init.name, weight.shape)
        tmp_in = helper.make_tensor_value_info(init.name, TensorProto.FLOAT, weight.shape)
        inits.append(tmp_in)
    graph.input.extend(inits)

## node
def print_all_node(graph):
    ## node: have name(op name), input(input id), output(input id), op_type, attribute
    # print("node: ", graph.node)
    for i, n in enumerate(graph.node):
        print("node id:", i)
        print("node name:", n.name)
        print("node type:", n.op_type)
        print("input:", n.input)
        print("output:", n.output)
        print("attriburte:", n.attribute)
        print()

def print_node(graph, op_name):
    ## node: have name(op name), input(input id), output(input id), op_type, attribute
    # print("node: ", graph.node)
    for i, n in enumerate(graph.node):
        if n.name==op_name:
            print("node id:", i)
            print("node name:", n.name)
            print("node type:", n.op_type)
            print("input:", n.input)
            print("output:", n.output)
            print("attriburte:", n.attribute)
            print()

def get_node_id(graph, node_name):
    for i, n in enumerate(graph.node):
        if n.name==node_name:
            return i
    print("node", node_name, "not found")
    return None

def insert_node(graph, node_id, node):
    graph.node.insert(node_id, node)

def remove_node(graph, node_id):
    node = graph.node[node_id]
    graph.node.remove(node)

def edit_node(graph):
    ## edit node (Resize)
    node_id = print_node(graph, "Resize")
    graph.node[node_id].attribute.remove(graph.node[node_id].attribute[0])
    tmp_attr1 = helper.make_attribute("coordinate_transformation_mode", "half_pixel")
    # tmp_attr2 = helper.make_attribute("cubic_coeff_a", -0.75)
    tmp_attr3 = helper.make_attribute("mode", "nearest")
    tmp_attr4 = helper.make_attribute("nearest_mode", "floor")
    # graph.node[node_id].attribute.extend([tmp_attr1, tmp_attr2, tmp_attr3, tmp_attr4])
    graph.node[node_id].attribute.extend([tmp_attr1, tmp_attr3, tmp_attr4])
    print(graph.node[node_id].attribute)

    node_id = print_node(graph, "Resize1")
    graph.node[node_id].attribute.remove(graph.node[node_id].attribute[0])
    tmp_attr1 = helper.make_attribute("coordinate_transformation_mode", "half_pixel")
    # tmp_attr2 = helper.make_attribute("cubic_coeff_a", -0.75)
    tmp_attr3 = helper.make_attribute("mode", "nearest")
    tmp_attr4 = helper.make_attribute("nearest_mode", "floor")
    # graph.node[node_id].attribute.extend([tmp_attr1, tmp_attr2, tmp_attr3, tmp_attr4])
    graph.node[node_id].attribute.extend([tmp_attr1, tmp_attr3, tmp_attr4])
    print(graph.node[node_id].attribute)

def create_node(graph):
       
    inputs = ["onnx::Sigmoid_19", "Resize_roi", "Resize_scales"]
    outputs = ["Resize_output"]
    tmp_attr1 = helper.make_attribute("coordinate_transformation_mode", "asymmetric")
    tmp_attr2 = helper.make_attribute("cubic_coeff_a", -0.75)
    tmp_attr3 = helper.make_attribute("mode", "nearest")
    tmp_attr4 = helper.make_attribute("nearest_mode", "floor")
    node = onnx.helper.make_node("Resize", inputs, outputs, "Resize1")
    print("ori attr:", node.attribute)
    node.attribute.extend([tmp_attr1, tmp_attr2, tmp_attr3, tmp_attr4])
    print("attr:", node.attribute)
    return node

## other
def edit_padding(graph):
    ## node: have name(op name), input(input id), output(input id), op_type, attribute
    # print("node: ", graph.node)
    for i, n in enumerate(graph.node):
        if n.op_type=="Conv" and n.attribute[0].name=="auto_pad" and n.attribute[3].name=="kernel_shape":
            pad_type = n.attribute[0].s.decode('UTF-8')
            kernel = n.attribute[3].ints
            ## make new attributes
            if pad_type=="SAME_UPPER":
                tmp_attr1 = helper.make_attribute("auto_pad", "NOTSET")
                graph.node[i].attribute.remove(n.attribute[0])
                if kernel[0]==3:
                    tmp_attr2 = helper.make_attribute("pads", [1,1,1,1])
                else:
                    tmp_attr2 = helper.make_attribute("pads", [0,0,0,0])
                graph.node[i].attribute.extend([tmp_attr1, tmp_attr2])

def del_branch(graph, start_id, end_id):
    print("delete node between", start_id, "~", end_id)
    for i in range(end_id, start_id-1, -1):
        remove_node(graph, i)    

def remove_attribute(graph, removed_attr):
    for i, n in enumerate(graph.node):
        for j, attr in enumerate(n.attribute):
            if attr.name==removed_attr:
                print("Remove "+n.name+" attribute "+removed_attr)
                graph.node[i].attribute.remove(graph.node[i].attribute[j])


""" load model """
model = onnx.load("yolov5m.onnx")
graph = model.graph
# print_all_node(graph)
# print_node(graph, "Conv_0")
# print_all_initializer(graph)
# print_input(graph)

# """ edit output"""
# for _ in range(4):
#     graph.output.remove(graph.output[-1])
# graph.output.insert()
# graph.output[0].name = "443"
# graph.output[1].name = "501"
# graph.output[2].name = "559"



# """ delete nodes """
# ## delete node
# for node_name in ["Concat_421"]:
#     node_id = get_node_id(graph, node_name)
#     print("delete node", node_id)
#     if node_id:
#         remove_node(graph, node_id)

# ## delete branch 3
# branch_start = get_node_id(graph, "Reshape_385")
# branch_end = get_node_id(graph, "Reshape_420")
# del_branch(graph, branch_start, branch_end)

# ## delete branch 2
# branch_start = get_node_id(graph, "Reshape_335")
# branch_end = get_node_id(graph, "Reshape_370")
# del_branch(graph, branch_start, branch_end)

# ## delete branch 1
# branch_start = get_node_id(graph, "Reshape_285")
# branch_end = get_node_id(graph, "Reshape_320")
# del_branch(graph, branch_start, branch_end)

""" make new model """
# new_graph = helper.make_graph(graph.node, 'yolov5m', graph.input, [graph.node[384], graph.node[334], graph.node[284]])
# new_model = helper.make_model(new_graph)

## change ir version
model.ir_version = 7
## sometimes change version need add initializer to input
# add_initializer_to_input(model.graph)
## convert opset version
model = version_converter.convert_version(model , 12)

model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)


""" save model """
# onnx.save(model, "yolov5m_del_post.onnx")
onnx.utils.extract_model("yolov5m.onnx", "yolov5m_del_post.onnx", ["images"], ["518", "576", "460"])