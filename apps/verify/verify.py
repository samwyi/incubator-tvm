# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _verify correctness and performance of a tuned model:

====================
**Author**: `Yi Wang <https://github.com/samwyi>`_

Usage:
    python3 verify.py --config example.config

The config file describes:
    Input:
        (1) the model: can be TVM bulidin models, mxnet models or custom defined model
        (2) the data
        (3) ground truth result to compare
        (4) multiple tuned logs to compare
    Output:
        output file name

An example config file is verify.config
"""
import tvm
import tvm.relay as relay
import numpy as np
from tvm import autotvm
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from PIL import Image
from matplotlib import pyplot as plt
import time
import argparse
import configparser
from tvm.contrib.download import download_testdata
import logging

logging.getLogger('autotvm').setLevel(logging.ERROR)

#################################################################
# get network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

def get_network(source, name, dtype, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if 'mxnet' in source:
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model(name, pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        ## we want a probability so add a softmax operator
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    elif 'relay' in source:
        if "resnet" in name:
            n_layer = int(name.split('-')[1])
            mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif "vgg" in name:
            n_layer = int(name.split('-')[1])
            mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
        elif name == 'mobilenet':
            mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
        elif name == 'squeezenet_v1.1':
            mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
        elif name == 'inception_v3':
            input_shape = (1, 3, 299, 299)
            mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        else:
            raise ValueError("Unsupported network: " + name)
    else:
        raise ValueError("unsupported source: " + source)
    return mod, params, input_shape, output_shape

######################################################################
# Parse Config and download data
def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def get_data(config):
    img_path = download_testdata(config['data']['url'], config['data']['file'], module='data')
    image = Image.open(img_path).resize((224, 224))
    # plt.imshow(image)
    # plt.show()
    x = transform_image(image)
    return x

def get_truth(config):
    synset_path = download_testdata(config['truth']['url'], config['truth']['file'], module='data')
    with open(synset_path) as f:
        synset = eval(f.read())
    return synset

######################################################################
# record verifcation environment
def record_env(config, output_file):
    record_sessions = ['model', 'data', 'device', 'run']
    output_file.write("\nVerification Environment:\n")
    for session in record_sessions:
        output_file.write("\n[%s]\n" % session)
        for name, value in config.items(session):
            output_file.write("%s = %s \n" % (name, value))

########################################################################
# evaluate the model with a given tune_log file
def evaluate(remote, pass_id, mod, params, input_shape, img, truth, config, tune_log, output_file):

    target = tvm.target.create( config['device']['target'] )

    # clear the cache in the compiler engine to make sure it rebuild the model from scratch
    relay.backend.compile_engine.get().clear()
    if not tune_log:
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target, params=params)
    else:
        with autotvm.apply_history_best(tune_log):
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod, target, params=params)

    # export library
    tmp = tempdir()
    if 'android' in config['device']['os']:
        from tvm.contrib import ndk
        filename = str(pass_id) + "net.so"
        lib.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = str(pass_id) + "net.tar"
        lib.export_library(tmp.relpath(filename))

    # upload module to device
    remote.upload(tmp.relpath(filename))
    rlib = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)
    module = runtime.create(graph, rlib, ctx)

    data_tvm = tvm.nd.array(img.astype(config['model']['dtype']))
    module.set_input('data', data_tvm)
    module.set_input(**params)

    ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=int(config['run']['repeat']))
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond

    # get outputs
    tvm_output = module.get_output(0)
    top1 = np.argmax(tvm_output.asnumpy()[0])
    result = ("%-20s %-15r %-10s (%s)    %s" % ("pass %d" % pass_id,
                                                 truth[top1],
                                                 "%.2f ms" % np.mean(prof_res),
                                                 "%.2f ms" % np.std(prof_res),
                                                 tune_log if tune_log else "default config" ))
    output_file.write(result + '\n')
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='verify.config')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    img = get_data(config)
    truth = get_truth(config)

    mod, params, input_shape, _ = get_network(config['model']['source'],
                                              config['model']['name'],
                                              config['model']['dtype'],
                                              batch_size = 1)

    passes = config.items('compare')

    remote = autotvm.measure.request_remote(config['device']['key'],
                                            config['device']['ip'],
                                            int(config['device']['port']),
                                            timeout=10000)

    with open(config['run']['output'], 'w') as output_file:
        header = ("%-20s %-15s %-10s (%s)    %s" % (config['model']['name'],
                                                 'result',
                                                 'mean',
                                                 'std dev',
                                                 'config'))
        output_file.write(header + '\n')
        print(header)

        for i, (_, tune_log) in enumerate(passes):
            evaluate(remote, i, mod, params, input_shape, img, truth, config, tune_log, output_file)

        record_env(config, output_file)
